# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    # For this task q=3, n=7, radius=1.
    # We implement a greedy set-cover style construction with local search improvement.
    # Deadline guard: 140s (10s safety margin before 150s kill).
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: direct product construction
        # For general case, use a simple greedy
        return _greedy_cover(q, n, radius)
    
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    # Precompute all 2187 words as tuples
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)  # 2187
    
    # For each word, precompute the set of words within Hamming distance 1 (its "ball")
    # ball[w] = set of indices of words within distance <= 1 from w
    # We need this for coverage checks.
    # Actually, for set cover: each candidate codeword c covers word w if dist(c,w) <= radius.
    # So for each candidate c, the set of words it covers is ball(c).
    # We'll precompute ball for each word index.
    
    def neighbors(w):
        """Yield all words within Hamming distance <= radius from w."""
        w = list(w)
        yield tuple(w)
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig:
                    continue
                w[i] = v
                yield tuple(w)
            w[i] = orig
    
    # Precompute coverage sets: for each candidate codeword index, which words it covers
    # Since every word is a potential codeword, coverage[i] = set of j such that dist(word_i, word_j) <= 1
    coverage = []
    for w in all_words:
        cov = set()
        for nw in neighbors(w):
            cov.add(word_to_idx[nw])
        coverage.append(cov)
    
    # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words
    uncovered = set(range(N))
    selected = []
    
    # For efficiency, track how many selected codewords cover each word
    # Greedy: at each step, pick codeword with max |coverage[c] ∩ uncovered|
    # To speed up, we can use a priority approach
    
    while uncovered and time.time() < DEADLINE - 20:
        best_c = -1
        best_count = -1
        # Iterate over candidates not yet selected (for efficiency, skip already selected)
        # For large N this is expensive; use a smarter approach
        # Actually with N=2187, iterating all is fine initially but gets slow
        # Let's track selected as a set for O(1) lookup
        selected_set = set(selected)
        for c in range(N):
            if c in selected_set:
                continue
            # Count how many uncovered words c covers
            cnt = 0
            for w in coverage[c]:
                if w in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_c = c
            if best_count == len(uncovered):
                break
        if best_c == -1 or best_count <= 0:
            break
        selected.append(best_c)
        selected_set.add(best_c)
        uncovered -= coverage[best_c]
    
    if not uncovered:
        return [list(all_words[c]) for c in selected]
    
    # If greedy didn't finish or left uncovered, do local search to improve
    # Local search: try swapping codewords to reduce total count while maintaining coverage
    # Also try removing redundant codewords
    
    # First, try to remove redundant codewords
    def is_redundant(c, selected_list, cov_data):
        """Check if codeword c is redundant given the other selected codewords."""
        # c is redundant if every word covered by c is also covered by some other selected codeword
        for w in cov_data[c]:
            covered = False
            for c2 in selected_list:
                if c2 == c:
                    continue
                if w in cov_data[c2]:
                    covered = True
                    break
            if not covered:
                return False
        return True
    
    # Remove redundant codewords iteratively
    changed = True
    while changed and time.time() < DEADLINE - 10:
        changed = False
        for c in list(selected):
            if is_redundant(c, selected, coverage):
                selected.remove(c)
                changed = True
    
    # Local search: try to replace one codeword with another that allows removal of another
    # Simpler: try pairwise swaps
    def try_improve(selected_list, cov_data, deadline):
        """Try to reduce the number of codewords by local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            sel = list(selected_list)
            for i in range(len(sel)):
                for j in range(N):
                    if j in sel:
                        continue
                    # Try replacing sel[i] with j
                    new_sel = [x for x in sel if x != sel[i]] + [j]
                    # Check if all words are still covered
                    # Build coverage set
                    cov_set = set()
                    for c in new_sel:
                        cov_set |= cov_data[c]
                    if len(cov_set) == N:
                        selected_list[:] = new_sel
                        improved = True
                        break
                if improved:
                    break
        return selected_list
    
    # The pairwise swap above is O(N^2 * N) which is too slow.
    # Better approach: use the ball structure.
    # For each selected codeword c, find words that ONLY c covers (critical words).
    # If a codeword has no critical words, it's redundant (already handled).
    # To remove a codeword, we need to add codewords that cover its critical words.
    
    # Alternative: kick-restart approach from the knowledge
    # Remove a few codewords, then re-greedy to cover the gaps
    
    def kick_and_regreedy(selected_list, cov_data, num_remove, deadline):
        """Remove num_remove random codewords, then greedy to cover uncovered."""
        if len(selected_list) <= num_remove:
            return list(selected_list)
        
        # Find words that are covered by exactly one selected codeword
        # These are critical for that codeword
        word_cover_count = {}
        for w in range(N):
            count = 0
            for c in selected_list:
                if w in cov_data[c]:
                    count += 1
            word_cover_count[w] = count
        
        # Critical words for each selected codeword
        critical = {}
        for c in selected_list:
            crit = set()
            for w in cov_data[c]:
                if word_cover_count[w] == 1:
                    crit.add(w)
            critical[c] = crit
        
        # Pick codewords with fewest critical words to remove
        candidates = sorted(selected_list, key=lambda c: len(critical[c]))
        to_remove = candidates[:num_remove]
        
        new_sel = [c for c in selected_list if c not in to_remove]
        
        # Find uncovered words
        covered = set()
        for c in new_sel:
            covered |= cov_data[c]
        uncovered = set(range(N)) - covered
        
        # Greedy to cover uncovered
        while uncovered and time.time() < deadline:
            best_c = -1
            best_count = -1
            for c in range(N):
                if c in new_sel:
                    continue
                cnt = len(coverage[c] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c == -1 or best_count <= 0:
                break
            new_sel.append(best_c)
            uncovered -= coverage[best_c]
        
        return new_sel
    
    # Try kick-and-regreedy with different removal counts
    best_selected = list(selected)
    best_len = len(best_selected)
    
    for num_remove in [1, 2, 3, 4, 5]:
        if time.time() > DEADLINE - 5:
            break
        # Try multiple random kicks
        for trial in range(10):
            if time.time() > DEADLINE - 5:
                break
            # Randomize which codewords to remove
            sel_copy = list(selected)
            random.shuffle(sel_copy)
            to_remove = sel_copy[:num_remove]
            
            new_sel = [c for c in sel_copy if c not in to_remove]
            
            covered = set()
            for c in new_sel:
                covered |= coverage[c]
            uncovered = set(range(N)) - covered
            
            # Greedy to cover uncovered
            while uncovered and time.time() < DEADLINE - 5:
                best_c = -1
                best_count = -1
                for c in range(N):
                    if c in new_sel:
                        continue
                    cnt = 0
                    for w in coverage[c]:
                        if w in uncovered:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_c = c
                if best_c == -1 or best_count <= 0:
                    break
                new_sel.append(best_c)
                uncovered -= coverage[best_c]
            
            if len(new_sel) < best_len:
                best_selected = new_sel
                best_len = len(best_selected)
                selected = list(best_selected)
    
    # Final redundancy removal
    changed = True
    while changed and time.time() < DEADLINE - 2:
        changed = False
        for c in list(selected):
            if is_redundant(c, selected, coverage):
                selected.remove(c)
                changed = True
    
    return [list(all_words[c]) for c in selected]


def _greedy_cover(q, n, radius):
    """General greedy set cover for arbitrary q, n, radius."""
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    # Greedy
    uncovered = set(range(N))
    selected = []
    
    while uncovered:
        best_c = -1
        best_count = -1
        for c in range(N):
            if c in selected:
                continue
            cnt = 0
            for w in uncovered:
                if hamming_dist(all_words[c], all_words[w]) <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_c = c
        if best_c == -1:
            break
        selected.append(best_c)
        to_remove = set()
        for w in list(uncovered):
            if hamming_dist(all_words[best_c], all_words[w]) <= radius:
                to_remove.add(w)
        uncovered -= to_remove
    
    return [list(all_words[c]) for c in selected]
# EVOLVE-BLOCK-END
