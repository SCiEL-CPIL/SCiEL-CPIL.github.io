# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _greedy_cover(balls_np, total_words, deadline):
    covered = np.zeros(total_words, dtype=np.int32)
    chosen = []
    uncovered_count = total_words
    while uncovered_count > 0 and time.time() < deadline - 5:
        gains = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            gains[i] = np.sum(covered[balls_np[i]] == 0)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        chosen.append(best_idx)
        for idx in balls_np[best_idx]:
            if covered[idx] == 0:
                uncovered_count -= 1
            covered[idx] += 1
    return chosen, covered, uncovered_count


def _local_search(chosen, balls_np, balls_sets, word_index, total_words, covered, deadline):
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Redundancy removal
        for i in range(len(chosen)):
            if time.time() > deadline - 2:
                break
            ball_i = balls_sets[chosen[i]]
            redundant = True
            for v in ball_i:
                if covered[word_index[v]] <= 1:
                    redundant = False
                    break
            if redundant:
                for v in ball_i:
                    covered[word_index[v]] -= 1
                chosen.pop(i)
                improved = True
                break
        # Swap
        if not improved:
            for i in range(len(chosen)):
                if time.time() > deadline - 2:
                    break
                ball_i = balls_sets[chosen[i]]
                lost = [word_index[v] for v in ball_i if covered[word_index[v]] == 1]
                if not lost:
                    for v in ball_i:
                        covered[word_index[v]] -= 1
                    chosen.pop(i)
                    improved = True
                    break
                lost_set = set(lost)
                best_replace = None
                best_new_gain = 0
                for w_idx in range(total_words):
                    if w_idx in chosen:
                        continue
                    ball_w = balls_sets[w_idx]
                    covers_all = all(idx in ball_w for idx in lost)
                    if not covers_all:
                        continue
                    new_gain = 0
                    for v in ball_w:
                        idx = word_index[v]
                        if covered[idx] == 0 and idx not in lost_set:
                            new_gain += 1
                    if new_gain > best_new_gain:
                        best_new_gain = new_gain
                        best_replace = w_idx
                if best_replace is not None:
                    for v in ball_i:
                        covered[word_index[v]] -= 1
                    chosen[i] = best_replace
                    for v in balls_sets[best_replace]:
                        covered[word_index[v]] += 1
                    improved = True
                    break
    return chosen, covered


def _branch_and_bound(chosen, covered, balls_np, balls_sets, word_index, total_words, deadline):
    # Fractional LP dual lower bound
    # max sum(y_j) s.t. for each set i: sum_{j in ball_i} y_j <= 1, y_j >= 0
    # Greedy approximation: repeatedly pick uncovered word with fewest candidate sets, assign 1/deg
    # But for speed, use a simpler bound: max number of uncovered words / max ball size (15)
    # Actually, use a packing-based bound:
    # We want to find a set of words that are pairwise at distance > 2 (so their balls are disjoint)
    # The size of such a packing is a lower bound on the number of additional codewords needed.
    # Use greedy packing: pick uncovered word, add to packing, remove all words within distance 2.
    
    best_chosen = list(chosen)
    best_size = len(chosen)
    
    uncovered = np.where(covered == 0)[0]
    if len(uncovered) == 0:
        return best_chosen
    
    # Greedy packing lower bound
    def packing_lower_bound(uncovered_mask):
        if not np.any(uncovered_mask):
            return 0
        packing = 0
        remaining = uncovered_mask.copy()
        while np.any(remaining):
            # Pick first remaining word
            idx = int(np.argmax(remaining))
            packing += 1
            # Remove all words within distance 2 from idx
            # Words within distance 2 from idx: all words that share at least 5 positions with idx
            # This is expensive to compute directly. Use a precomputed structure or approximate.
            # For speed, just remove idx itself and its neighbors (distance 1)
            # This gives a weaker but faster bound
            w = np.zeros(total_words, dtype=np.int32)
            w[idx] = 1
            for pos in range(7):
                for val in range(3):
                    new_idx = idx
                    # Convert idx to word, modify, convert back
                    word = list(word_index_inv[idx])
                    if word[pos] != val:
                        word[pos] = val
                        new_idx = word_index[tuple(word)]
                        w[new_idx] = 1
            remaining[w == 1] = 0
        return packing
    
    # Precompute word_index_inv
    global word_index_inv
    word_index_inv = [None] * total_words
    for w, i in word_index.items():
        word_index_inv[i] = w
    
    # Simple branch and bound
    def bb(chosen_list, covered_arr, depth):
        nonlocal best_chosen, best_size
        if time.time() > deadline - 2:
            return
        if len(chosen_list) >= best_size:
            return
        
        uncovered_mask = covered_arr == 0
        if not np.any(uncovered_mask):
            if len(chosen_list) < best_size:
                best_chosen = list(chosen_list)
                best_size = len(chosen_list)
            return
        
        # Lower bound: at least 1 more codeword needed
        lb = 1
        if len(chosen_list) + lb >= best_size:
            return
        
        # Pick the uncovered word with fewest candidate codewords that cover it
        # Candidate codewords for word w: all words within distance 1 of w
        # We want to branch on which codeword to add
        uncovered_indices = np.where(uncovered_mask)[0]
        
        # Pick the uncovered word with minimum number of uncovered neighbors (most constrained)
        best_uncovered = None
        best_degree = float('inf')
        for idx in uncovered_indices[:50]:  # Limit for speed
            # Count how many words in its ball are also uncovered
            ball = balls_np[idx]
            deg = np.sum(covered_arr[ball] == 0)
            if deg < best_degree:
                best_degree = deg
                best_uncovered = idx
        
        if best_uncovered is None:
            return
        
        # Branch: try adding each word that covers best_uncovered
        ball_of_uncovered = balls_np[best_uncovered]
        candidates = [int(x) for x in ball_of_uncovered if x not in chosen_list]
        
        # Sort by marginal gain
        candidates.sort(key=lambda x: -np.sum(covered_arr[balls_np[x]] == 0))
        
        for cand in candidates[:10]:  # Limit branching factor
            if time.time() > deadline - 2:
                return
            new_covered = covered_arr.copy()
            for idx in balls_np[cand]:
                new_covered[idx] += 1
            bb(chosen_list + [cand], new_covered, depth + 1)
    
    bb(list(chosen), covered.copy(), 0)
    return best_chosen


def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    total_words = 3 ** 7  # 2187

    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(3), repeat=7)]
    word_index = {w: i for i, w in enumerate(all_words)}
    
    # Precompute word_index_inv
    word_index_inv = [None] * total_words
    for w, i in word_index.items():
        word_index_inv[i] = w

    # For each word, precompute its ball (all words within distance 1)
    balls = {}
    for w in all_words:
        ball = set()
        ball.add(w)
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.add(tuple(new_w))
        balls[w] = ball
    
    # Convert to sets for local search
    balls_sets = {i: set(word_index[v] for v in balls[all_words[i]]) for i in range(total_words)}
    
    # Convert balls to numpy arrays for faster operations
    balls_np = np.array([[word_index[v] for v in balls[all_words[i]]] for i in range(total_words)], dtype=np.int32)

    # Greedy initialization
    chosen, covered, uncovered_count = _greedy_cover(balls_np, total_words, deadline)
    
    # Local search polish
    chosen, covered = _local_search(chosen, balls_np, balls_sets, word_index, total_words, covered, deadline)
    
    # Branch and bound refinement
    if time.time() < deadline - 10:
        chosen = _branch_and_bound(chosen, covered, balls_np, balls_sets, word_index, total_words, deadline)
    
    # Convert to lists
    return [list(all_words[i]) for i in chosen]
# EVOLVE-BLOCK-END
