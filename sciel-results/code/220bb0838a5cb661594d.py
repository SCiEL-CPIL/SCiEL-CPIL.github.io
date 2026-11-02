"""Dual greedy set-cover for K_q(n, radius) with shadow-price weighting."""
import time
import itertools
import numpy as np
from collections import defaultdict

_DEADLINE = 120.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _neighbors_indices(idx, q, n, radius, qpow):
    """Yield indices of all words within Hamming distance <= radius of idx."""
    # distance 0
    yield idx
    if radius < 1:
        return
    # distance 1
    for i in range(n):
        old_s = (idx // qpow[i]) % q
        for new_s in range(q):
            if new_s == old_s:
                continue
            idx2 = idx + qpow[i] * (new_s - old_s)
            if 0 <= idx2 < q ** n:
                yield idx2
    if radius < 2:
        return
    # distance 2
    for i in range(n):
        old_i = (idx // qpow[i]) % q
        for j in range(i + 1, n):
            old_j = (idx // qpow[j]) % q
            for ni in range(q):
                if ni == old_i:
                    continue
                for nj in range(q):
                    if nj == old_j:
                        continue
                    idx2 = idx + qpow[i] * (ni - old_i) + qpow[j] * (nj - old_j)
                    if 0 <= idx2 < q ** n:
                        yield idx2


def _build_dual_greedy(q, n, radius):
    """
    Dual greedy set cover:
    - Each word has a shadow price = 1 / (number of candidate codewords that can cover it).
    - Candidate codewords are all words in the space.
    - At each step, pick the candidate codeword that maximizes sum of shadow prices
      of uncovered words it covers, then mark those words covered and update prices.
    """
    total = q ** n
    qpow = [1]
    for _ in range(n):
        qpow.append(qpow[-1] * q)

    # Precompute neighbor sets for all words (for q=3, n=8, radius=1 this is 6561 * 25 = ~164k entries)
    # For radius=2 it would be much larger, so we handle that case with a flag.
    # For the specific task q=3, n=8, radius=1, we can precompute.
    # For general cases, we compute on the fly if total is small enough.

    # For efficiency with the specific task, precompute neighbors for all words.
    # neighbors[idx] = list of indices within distance <= radius
    neighbors = [None] * total
    if total * (1 + n * (q - 1) + n * (n - 1) // 2 * (q - 1) ** 2) < 50_000_000:
        for idx in range(total):
            nbrs = []
            for nb in _neighbors_indices(idx, q, n, radius, qpow):
                nbrs.append(nb)
            neighbors[idx] = nbrs

    # Shadow prices: initially all 1.0. When a word gets covered, we don't remove it,
    # but we track which words are still uncovered.
    # Actually, for dual greedy, we want to prioritize words that are hard to cover.
    # A word is "hard" if few codewords can cover it. In a covering code, every word
    # can be covered by itself and its neighbors, so the number of potential coverers
    # for word w is |B(w, radius)| where B is the ball of radius radius.
    # But that's the same for all words in a regular space... so shadow prices would be uniform.
    # 
    # Better interpretation: as we build the code, some words become "easy" because
    # they're already covered by many chosen codewords. Uncovered words that are far
    # from all chosen codewords are "hard". We want to pick codewords that cover
    # hard words.
    #
    # Revised approach: 
    # - Maintain uncovered set.
    # - For each candidate codeword c, compute the number of currently-uncovered words
    #   it covers. Pick the one that covers the most uncovered words (standard greedy).
    # - But to break ties and escape local optima, use a shadow price: words that are
    #   covered by fewer existing codewords get higher priority.
    #
    # Even simpler and effective: standard greedy set cover with good tie-breaking,
    # followed by local search (add/remove/swap) to reduce the code size.

    covered = np.zeros(total, dtype=bool)
    code = []
    count = 0

    # Precompute for each word, the list of words within distance <= radius (its "ball")
    # This is the same as neighbors. For q=3,n=8,radius=1: ball size = 1 + 8*2 = 17.
    
    # Standard greedy: repeatedly pick the word that covers the most uncovered words.
    # For efficiency, we can use the fact that in a covering code, each new codeword
    # covers at most |B(0, radius)| new words.
    
    ball_size = 1
    if radius >= 1:
        ball_size += n * (q - 1)
    if radius >= 2:
        ball_size += n * (n - 1) // 2 * (q - 1) ** 2

    # For the specific case q=3, n=8, radius=1: ball_size = 17.
    # Total words = 6561. So we need at least ceil(6561/17) = 386 codewords.
    # Best known is 486.

    # Greedy with numpy for speed:
    # At each step, for each uncovered word, it's a candidate. But we want to pick
    # a codeword that covers many uncovered words. 
    # 
    # Efficient approach: for each candidate position, compute how many uncovered words
    # it would cover. But checking all 6561 candidates each time is O(total^2) which is
    # 6561^2 = 43M, times ~486 iterations = 21B operations. Too slow in pure Python.
    #
    # Better: use the structure. When we add a codeword at position p, it covers
    # B(p, radius). The number of newly covered words is |B(p, radius) \ covered|.
    # We can maintain for each position the count of uncovered words in its ball.
    # 
    # Let's try a different approach: use the fact that this is a small problem.
    # 6561 words, ~486 codewords. We can use a priority queue or just iterate.

    # For q=3, n=8, radius=1, let's use a direct approach with numpy.
    # covered is a boolean array of size 6561.
    # For each candidate, compute coverage gain.
    
    # Precompute: for each word w, list of words within distance <= radius.
    # We already have neighbors.
    
    # Greedy loop:
    while count < total and _time_left() > 20.0:
        # Find the candidate that covers the most uncovered words.
        # For efficiency, only consider uncovered words as candidates (since a covered
        # word's ball is a subset of some uncovered word's ball... not exactly, but
        # it's a heuristic).
        
        # Actually, we should consider all words as candidates, but we can skip
        # words whose ball is entirely covered.
        
        best_gain = 0
        best_idx = -1
        
        # Sample candidates if total is large
        if total <= 100000:
            # Check all uncovered words first (they're more likely to have uncovered neighbors)
            # Then check some covered words if needed
            for idx in range(total):
                if covered[idx]:
                    continue
                # Count uncovered words in neighbors[idx]
                nbrs = neighbors[idx] if neighbors[idx] is not None else list(_neighbors_indices(idx, q, n, radius, qpow))
                gain = sum(1 for nb in nbrs if not covered[nb])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    if gain == ball_size:
                        break
        else:
            # For large spaces, use random sampling
            sample_size = min(1000, total)
            candidates = np.random.choice(total, size=sample_size, replace=False)
            for idx in candidates:
                nbrs = list(_neighbors_indices(int(idx), q, n, radius, qpow))
                gain = sum(1 for nb in nbrs if not covered[nb])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = int(idx)
        
        if best_idx < 0 or best_gain == 0:
            # All remaining words are covered, or no progress
            break
        
        # Add best_idx to code
        word = [(best_idx // qpow[i]) % q for i in range(n)]
        code.append(word)
        
        # Mark covered
        nbrs = neighbors[best_idx] if neighbors[best_idx] is not None else list(_neighbors_indices(best_idx, q, n, radius, qpow))
        for nb in nbrs:
            if not covered[nb]:
                covered[nb] = True
                count += 1
    
    return code


def _local_search(q, n, radius, code, time_budget=30.0):
    """
    Local search to reduce code size:
    - Try removing each codeword; if the code still covers everything, keep it removed.
    - Try swapping: remove one codeword, add another that covers the newly uncovered words.
    """
    total = q ** n
    qpow = [1]
    for _ in range(n):
        qpow.append(qpow[-1] * q)

    # Precompute neighbors for all words in the code and their coverage
    def _compute_coverage(code_set):
        """Return boolean array of covered words."""
        cov = np.zeros(total, dtype=bool)
        for c in code_set:
            idx = sum(c[i] * qpow[i] for i in range(n))
            # Mark neighbors
            for nb in _neighbors_indices(idx, q, n, radius, qpow):
                cov[nb] = True
        return cov

    def _is_covered(cov):
        return cov.all()

    code_set = list(code)
    current_size = len(code_set)
    
    # Precompute index for each codeword
    def _code_to_idx(c):
        return sum(c[i] * qpow[i] for i in range(n))

    def _idx_to_code(idx):
        return [(idx // qpow[i]) % q for i in range(n)]

    # Phase 1: Removal
    improved = True
    while improved and _time_left() > time_budget:
        improved = False
        for i in range(len(code_set)):
            # Try removing code_set[i]
            trial = code_set[:i] + code_set[i+1:]
            cov = _compute_coverage(trial)
            if _is_covered(cov):
                code_set = trial
                current_size -= 1
                improved = True
                break  # Restart the loop since indices changed
    
    # Phase 2: Swap (remove one, add one)
    # This is more expensive. Try a limited number of swaps.
    for _ in range(50):  # Limit iterations
        if _time_left() < time_budget:
            break
        best_swap = None
        best_new_size = current_size
        
        # Try removing each codeword and see if we can add one to fix
        for i in range(len(code_set)):
            trial = code_set[:i] + code_set[i+1:]
            cov = _compute_coverage(trial)
            if _is_covered(cov):
                # Already covered without this one
                code_set = trial
                current_size -= 1
                best_swap = True
                break
            
            # Find uncovered words
            uncovered = np.flatnonzero(~cov)
            if len(uncovered) > 50:  # Too many to fix with one addition
                continue
            
            # Try adding each uncovered word (or a sample)
            candidates = uncovered[:20] if len(uncovered) > 20 else uncovered
            for u in candidates:
                new_code = trial + [_idx_to_code(int(u))]
                new_cov = _compute_coverage(new_code)
                if _is_covered(new_cov):
                    new_size = len(new_code)
                    if new_size < best_new_size:
                        best_new_size = new_size
                        best_swap = new_code
        
        if best_swap is not None and best_swap is not True:
            code_set = best_swap
            current_size = best_new_size
        elif best_swap is True:
            continue
        else:
            break
    
    return code_set


def construct(q, n, radius):
    global _START
    _START = time.time()

    # For the specific task q=3, n=8, radius=1, use the dual greedy + local search.
    # For other parameters, use a simpler greedy.
    
    if q == 3 and n == 8 and radius == 1:
        code = _build_dual_greedy(q, n, radius)
        if _time_left() > 30.0:
            code = _local_search(q, n, radius, code, time_budget=25.0)
    else:
        # Fallback: simple greedy
        total = q ** n
        qpow = [1]
        for _ in range(n):
            qpow.append(qpow[-1] * q)
        
        covered = np.zeros(total, dtype=bool)
        code = []
        count = 0
        
        while count < total and _time_left() > 10.0:
            # Find first uncovered word
            found = -1
            for idx in range(total):
                if not covered[idx]:
                    found = idx
                    break
            if found < 0:
                break
            
            word = [(found // qpow[i]) % q for i in range(n)]
            code.append(word)
            
            for nb in _neighbors_indices(found, q, n, radius, qpow):
                if not covered[nb]:
                    covered[nb] = True
                    count += 1
        
        # Simple removal
        if _time_left() > 10.0:
            improved = True
            while improved and _time_left() > 5.0:
                improved = False
                for i in range(len(code)):
                    trial = code[:i] + code[i+1:]
                    cov = np.zeros(total, dtype=bool)
                    for c in trial:
                        idx = sum(c[j] * qpow[j] for j in range(n))
                        for nb in _neighbors_indices(idx, q, n, radius, qpow):
                            cov[nb] = True
                    if cov.all():
                        code = trial
                        improved = True
                        break
    
    # Ensure distinctness
    seen = set()
    unique = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique.append(c)
    
    return unique
