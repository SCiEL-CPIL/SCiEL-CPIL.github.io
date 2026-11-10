"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


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

    # Precompute balls as index arrays
    # Ball size: 1 + 7*2 = 15
    balls_indices = np.zeros((total_words, 15), dtype=np.int32)
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(word_index[tuple(new_w)])
        balls_indices[i] = ball

    # Build adjacency: for each word, which codewords cover it (i.e., are in its ball)
    # adj[w] = list of codeword indices c such that w is in ball(c)
    # This is the transpose of balls_indices
    adj = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls_indices[c]:
            adj[w].append(c)

    # Greedy set cover using vectorized gain computation
    covered = np.zeros(total_words, dtype=np.int8)
    chosen = []
    uncovered_count = total_words

    while uncovered_count > 0 and time.time() < deadline - 40:
        # Compute gain: for each candidate c, count uncovered words in its ball
        # Vectorized: gains[c] = sum(covered[balls_indices[c]] == 0)
        gains = np.array([np.sum(covered[balls_indices[c]] == 0) for c in range(total_words)], dtype=np.int32)
        
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
        
        chosen.append(best_idx)
        
        # Update coverage
        ball_indices = balls_indices[best_idx]
        for idx in ball_indices:
            if covered[idx] == 0:
                uncovered_count -= 1
            covered[idx] += 1

    # Local optimization phase
    def try_remove(chosen, covered, balls_indices, i):
        """Try to remove codeword i if redundant. Returns True if removed."""
        ball_i = balls_indices[chosen[i]]
        for idx in ball_i:
            if covered[idx] <= 1:
                return False
        for idx in ball_i:
            covered[idx] -= 1
        chosen.pop(i)
        return True

    def try_swap(chosen, covered, balls_indices, i):
        """Try to replace codeword i with a better one. Returns True if swapped."""
        ball_i = balls_indices[chosen[i]]
        lost = [idx for idx in ball_i if covered[idx] == 1]
        
        if not lost:
            # Redundant
            for idx in ball_i:
                covered[idx] -= 1
            chosen.pop(i)
            return True
        
        # Find replacement that covers all lost words
        candidates = set(balls_indices[lost[0]].tolist())
        for l in lost[1:]:
            candidates &= set(balls_indices[l].tolist())
        
        best_replace = None
        best_gain = 0
        for j in candidates:
            if j in chosen:
                continue
            ball_j = balls_indices[j]
            gain = np.sum(covered[ball_j] == 0)
            if gain > best_gain:
                best_gain = gain
                best_replace = j
        
        if best_replace is not None:
            for idx in ball_i:
                covered[idx] -= 1
            chosen[i] = best_replace
            for idx in balls_indices[best_replace]:
                covered[idx] += 1
            return True
        return False

    # Iterative local optimization
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        n_chosen = len(chosen)
        
        # Pass 1: Try removing redundant codewords
        for i in range(n_chosen - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            if try_remove(chosen, covered, balls_indices, i):
                improved = True
                break
        
        # Pass 2: Try 1-for-1 swaps
        if not improved:
            for i in range(len(chosen)):
                if time.time() > deadline - 10:
                    break
                if try_swap(chosen, covered, balls_indices, i):
                    improved = True
                    break

    # Simulated annealing polish phase (Simplified for robustness)
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        best_size = len(chosen)
        best_chosen = list(chosen)
        best_covered = covered.copy()
        
        T = 2.0
        T_min = 0.1
        cooling = 0.99
        
        while T > T_min and time.time() < deadline - 5:
            n_chosen = len(chosen)
            if n_chosen == 0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try removing a random codeword
                i = int(rng.integers(0, n_chosen))
                c = chosen[i]
                ball_c = balls_indices[c]
                uncovered_after = int(np.sum(covered[ball_c] == 1))
                
                if uncovered_after == 0:
                    for idx in ball_c:
                        covered[idx] -= 1
                    chosen.pop(i)
                    if len(chosen) < best_size:
                        best_size = len(chosen)
                        best_chosen = list(chosen)
                        best_covered = covered.copy()
                else:
                    if rng.random() < np.exp(-uncovered_after / T):
                        for idx in ball_c:
                            covered[idx] -= 1
                        chosen.pop(i)
                        if len(chosen) < best_size:
                            best_size = len(chosen)
                            best_chosen = list(chosen)
                            best_covered = covered.copy()
                        else:
                            # Revert
                            for idx in ball_c:
                                covered[idx] += 1
                            chosen.insert(i, c)
            else:
                # Try swapping a random codeword
                i = int(rng.integers(0, n_chosen))
                c = chosen[i]
                ball_c = balls_indices[c]
                lost = [idx for idx in ball_c if covered[idx] == 1]
                
                if lost:
                    candidates = set(balls_indices[lost[0]].tolist())
                    for l in lost[1:]:
                        candidates &= set(balls_indices[l].tolist())
                    candidates.difference_update(chosen)
                    
                    if candidates:
                        cand_list = list(candidates)
                        j = cand_list[int(rng.integers(0, len(cand_list)))]
                        ball_j = balls_indices[j]
                        
                        old_covered = covered.copy()
                        for idx in ball_c:
                            covered[idx] -= 1
                        for idx in ball_j:
                            covered[idx] += 1
                        
                        new_uncovered = int(np.sum(covered == 0))
                        old_uncovered = int(np.sum(old_covered == 0))
                        
                        if new_uncovered <= old_uncovered:
                            chosen[i] = j
                            if len(chosen) < best_size:
                                best_size = len(chosen)
                                best_chosen = list(chosen)
                                best_covered = covered.copy()
                        else:
                            if rng.random() < np.exp(-(new_uncovered - old_uncovered) / T):
                                chosen[i] = j
                            else:
                                covered = old_covered
                else:
                    # Redundant, remove
                    for idx in ball_c:
                        covered[idx] -= 1
                    chosen.pop(i)
                    if len(chosen) < best_size:
                        best_size = len(chosen)
                        best_chosen = list(chosen)
                        best_covered = covered.copy()
            
            T *= cooling
        
        # Restore best solution found
        if best_size < len(chosen):
            chosen = best_chosen
            covered = best_covered

    # Final Repair Phase: Ensure validity
    # Check if any words are uncovered
    while np.any(covered == 0):
        if time.time() > deadline - 1:
            break
        uncovered_mask = (covered == 0)
        # Find codeword that covers the most uncovered words
        # gains[c] = sum(uncovered_mask[balls_indices[c]])
        gains = np.array([np.sum(uncovered_mask[balls_indices[c]]) for c in range(total_words)], dtype=np.int32)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break # Should not happen if valid
        chosen.append(best_idx)
        ball_indices = balls_indices[best_idx]
        for idx in ball_indices:
            if covered[idx] == 0:
                pass # It will be incremented
            covered[idx] += 1

    # Convert to lists
    return [list(all_words[idx]) for idx in chosen]
# EVOLVE-BLOCK-END
