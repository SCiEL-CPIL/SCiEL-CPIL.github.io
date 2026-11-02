import time
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallback: direct sum construction for general case
    return _fallback(q, n, radius)


def _solve_ternary_8_1(deadline):
    import numpy as np
    import time
    
    q, n, r = 3, 8, 1
    total = q ** n  # 6561
    
    # Precompute all words as indices
    # Each word is an index 0..6560, converted to base-3 digits
    
    def idx_to_word(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
    
    def word_to_idx(digits):
        idx = 0
        for i in range(n - 1, -1, -1):
            idx = idx * q + digits[i]
        return idx
    
    # Precompute neighbors for each word (within Hamming distance <= radius)
    # For radius 1, q=3, n=8: 1 + 8*2 = 17 neighbors per word
    # But we need: for each potential codeword position, which words it covers
    # Actually, let's precompute: for each codeword c, the set of words within distance r
    
    # Strategy: We'll work with all 6561 possible positions as potential codewords.
    # For each position c, compute which words it covers (c itself + neighbors at distance 1)
    
    # Precompute the coverage mapping: cov[c] = list of word indices covered by codeword at position c
    # For radius 1, q=3, n=8:
    # c covers c and all words at Hamming distance 1 from c
    
    # To compute neighbors efficiently:
    # A word w is within distance 1 of c iff w differs from c in at most 1 position
    
    # Let's precompute for each word index, its neighbors within distance 1
    # Actually, we need: for each codeword position c, the set of covered words
    
    # Let's build this using the fact that:
    # covered_by(c) = {c} union {w : hamming(c,w) == 1}
    
    # For efficiency, precompute using digit manipulation
    
    # Precompute powers of q for index conversion
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q
    
    # Precompute digits for all words
    all_digits = np.zeros((total, n), dtype=np.uint8)
    for idx in range(total):
        tmp = idx
        for i in range(n):
            all_digits[idx, i] = tmp % q
            tmp //= q
    
    # For each codeword position c, compute covered word indices
    # cov[c] is a list of word indices within Hamming distance <= 1 from c
    cov = []
    for c in range(total):
        c_digits = all_digits[c]
        covered = [c]
        for pos in range(n):
            orig = c_digits[pos]
            for new_val in range(q):
                if new_val != orig:
                    # Create new word
                    tmp_idx = 0
                    for i in range(n):
                        if i == pos:
                            tmp_idx += new_val * q_powers[i]
                        else:
                            tmp_idx += c_digits[i] * q_powers[i]
                    covered.append(tmp_idx)
        cov.append(covered)
    
    # Now: dual greedy set cover
    # State: covered_count[word] = number of selected codewords that cover this word
    # We want all words to have covered_count >= 1
    
    # Shadow price: p[word] = 1 / (number of candidate codewords that can still cover word)
    # Initially, every word can be covered by |cov^{-1}(word)| codewords
    # cov^{-1}(word) = {c : word in cov[c]} = {c : hamming(c, word) <= 1}
    # For radius 1, q=3, n=8: each word is covered by 1 + 8*2 = 17 codewords
    
    # So initially, all words have the same number of potential coverers (17)
    # But as we select codewords, the "effective" scarcity changes
    
    # Let's think differently:
    # Maintain for each word w: remaining_coverers[w] = number of NOT-yet-selected codewords c such that w in cov[c]
    # When we select codeword c, for each w in cov[c], decrement remaining_coverers[w]
    # Shadow price p[w] = 1 / max(1, remaining_coverers[w])
    # Score of selecting codeword c = sum_{w in cov[c], covered_count[w]==0} p[w]
    #   + small_bonus * sum_{w in cov[c], covered_count[w]>0} p[w]  (to break ties)
    
    # Initialize
    covered_count = np.zeros(total, dtype=np.uint8)
    # remaining_coverers: for each word, how many potential codewords cover it
    # Initially all 17 (since each word has 17 neighbors including itself)
    remaining_coverers = np.full(total, 17, dtype=np.uint16)
    
    # Precompute reverse mapping: for each word w, which codewords cover it
    # rev[w] = list of codeword indices c such that w in cov[c]
    rev = [[] for _ in range(total)]
    for c in range(total):
        for w in cov[c]:
            rev[w].append(c)
    
    # Initially remaining_coverers[w] = len(rev[w]) = 17 for all w
    # Verify
    # (all are 17 for q=3, n=8, r=1)
    
    # Greedy selection
    selected = []
    selected_set = set()
    
    # For efficiency, maintain:
    # - uncovered_words: set of words with covered_count == 0
    # - For each codeword c, its score = sum of p[w] for w in cov[c] where w is uncovered
    # We need to update scores when we select a codeword
    
    # Shadow prices
    p = np.ones(total, dtype=np.float64)  # p[w] = 1/remaining_coverers[w]
    
    # For the greedy step, we need to find argmax_c score(c)
    # score(c) = sum_{w in cov[c]} p[w] * (1 if covered_count[w]==0 else 0)
    
    # This is expensive to recompute each time. Let's use a different approach:
    # At each step, for all not-yet-selected codewords, compute their score and pick the best.
    # With 6561 codewords and ~486 selections, this is 6561*486 ≈ 3.2M evaluations, each O(17)
    # Total: ~54M operations, should be fine in 140s
    
    # But we can optimize: only consider codewords that still have uncovered words to cover
    
    # Let's track: for each codeword c, how many of its covered words are still uncovered
    # useful_count[c] = |{w in cov[c] : covered_count[w] == 0}|
    
    useful_count = np.zeros(total, dtype=np.uint8)
    for c in range(total):
        useful_count[c] = len(cov[c])  # initially all 17 are uncovered
    
    # Also maintain score for each codeword
    # score[c] = sum_{w in cov[c], covered_count[w]==0} p[w]
    score = np.zeros(total, dtype=np.float64)
    for c in range(total):
        s = 0.0
        for w in cov[c]:
            if covered_count[w] == 0:
                s += p[w]
        score[c] = s
    
    # Greedy loop
    while True:
        if time.time() > deadline - 10:
            break
        
        # Find best codeword
        # Only consider codewords with useful_count > 0
        best_c = -1
        best_score = -1.0
        for c in range(total):
            if c in selected_set:
                continue
            if useful_count[c] == 0:
                continue
            if score[c] > best_score:
                best_score = score[c]
                best_c = c
        
        if best_c == -1 or best_score <= 0:
            break
        
        # Select best_c
        selected_set.add(best_c)
        selected.append(best_c)
        
        # Update covered_count, remaining_coverers, p, useful_count, score
        for w in cov[best_c]:
            old_cc = covered_count[w]
            covered_count[w] += 1
            
            # Update remaining_coverers and p
            remaining_coverers[w] -= 1
            if remaining_coverers[w] > 0:
                p[w] = 1.0 / remaining_coverers[w]
            else:
                p[w] = 0.0  # no more coverers; but this shouldn't happen if we're valid
            
            # Update useful_count for all codewords that cover w
            # But that's expensive. Instead, recompute scores lazily.
        
        # Recompute scores for all not-yet-selected codewords
        # This is O(total * 17) per selection = O(111K) per step
        # With ~500 selections: ~55M ops, OK
        
        # Actually, let's be smarter. Only update scores for codewords that cover any newly covered word.
        # For each w in cov[best_c], for each c in rev[w] (not selected), update score[c]
        
        # But this could be many updates. Let's just recompute all scores periodically.
        
        # For now, recompute all scores
        for c in range(total):
            if c in selected_set:
                continue
            s = 0.0
            uc = 0
            for w in cov[c]:
                if covered_count[w] == 0:
                    s += p[w]
                    uc += 1
            score[c] = s
            useful_count[c] = uc
        
        # Check if all covered
        # Count uncovered
        uncovered = np.sum(covered_count == 0)
        if uncovered == 0:
            break
    
    # Convert to codewords
    result = []
    for c in selected:
        digits = [0] * n
        tmp = c
        for i in range(n):
            digits[i] = tmp % q
            tmp //= q
        result.append(digits)
    
    # Local search: try to remove codewords
    if time.time() < deadline - 5:
        result = _local_search(result, q, n, radius, deadline)
    
    return result


def _local_search(codewords, q, n, radius, deadline):
    """Try to remove redundant codewords."""
    import numpy as np
    import time
    
    total = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q
    
    def word_to_idx(digits):
        idx = 0
        for i in range(n - 1, -1, -1):
            idx = idx * q + digits[i]
        return idx
    
    def get_covered(cw_idx):
        """Get list of word indices covered by codeword at position cw_idx."""
        digits = [0] * n
        tmp = cw_idx
        for i in range(n):
            digits[i] = tmp % q
            tmp //= q
        
        covered = [cw_idx]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val != orig:
                    tmp_idx = 0
                    for i in range(n):
                        if i == pos:
                            tmp_idx += new_val * q_powers[i]
                        else:
                            tmp_idx += digits[i] * q_powers[i]
                    covered.append(tmp_idx)
        return covered
    
    # Current set of codeword indices
    cw_indices = set()
    for cw in codewords:
        cw_indices.add(word_to_idx(cw))
    
    # For each word, count how many codewords cover it
    cover_count = np.zeros(total, dtype=np.uint8)
    cw_cover = {}
    for c in cw_indices:
        covered = get_covered(c)
        cw_cover[c] = covered
        for w in covered:
            cover_count[w] += 1
    
    # Try to remove codewords in order of lowest "uniqueness"
    # A codeword is removable if for all w in cov(c), cover_count[w] > 1
    
    removed = 0
    while time.time() < deadline - 3:
        progress = False
        for c in list(cw_indices):
            if time.time() > deadline - 3:
                break
            covered = cw_cover[c]
            removable = True
            for w in covered:
                if cover_count[w] <= 1:
                    removable = False
                    break
            if removable:
                # Remove c
                for w in covered:
                    cover_count[w] -= 1
                cw_indices.remove(c)
                del cw_cover[c]
                removed += 1
                progress = True
        if not progress:
            break
    
    # Build result
    result = []
    for c in cw_indices:
        digits = [0] * n
        tmp = c
        for i in range(n):
            digits[i] = tmp % q
            tmp //= q
        result.append(digits)
    
    return result


def _fallback(q, n, radius):
    import itertools
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Very basic fallback
    return [[0] * n]

