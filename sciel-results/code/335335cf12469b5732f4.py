import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s

    N = 51975

    # Generate all divisors of N greater than 1
    divisors = []
    for i in range(1, int(np.sqrt(N)) + 1):
        if N % i == 0:
            if i > 1:
                divisors.append(i)
            other = N // i
            if other != i and other > 1:
                divisors.append(other)
    divisors.sort()

    # Precompute index arrays for each divisor
    # idx_cache[m][r] = np.array of indices r, r+m, r+2m, ... < N
    idx_cache = {}
    for m in divisors:
        idx_cache[m] = [np.arange(r, N, m) for r in range(m)]

    def compute_score(classes):
        """Compute uncovered count from classes."""
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(covered == 0))

    def greedy_init(moduli_list):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        covered = np.zeros(N, dtype=np.uint8)
        classes = []
        for m in moduli_list:
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = int(np.sum(covered[r::m] == 0))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        uncovered = int(np.sum(covered == 0))
        return classes, uncovered

    def build_cover_count(classes):
        """Build cover_count array from classes."""
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in classes:
            cover_count[r::m] += 1
        return cover_count

    def fast_delta(cover_count, m, old_r, new_r):
        """Compute delta in uncovered count when changing residue from old_r to new_r for modulus m."""
        if old_r == new_r:
            return 0
        
        old_idx = idx_cache[m][old_r]
        new_idx = idx_cache[m][new_r]
        
        # Find elements in old but not in new, and vice versa
        old_set = set(old_idx)
        new_set = set(new_idx)
        
        only_old = old_set - new_set
        only_new = new_set - old_set
        
        newly_uncovered = int(np.sum(cover_count[list(only_old)] == 1)) if only_old else 0
        newly_covered = int(np.sum(cover_count[list(only_new)] == 0)) if only_new else 0
        
        return newly_uncovered - newly_covered

    def local_search(classes, score, time_limit, allow_two_swap=True):
        """
        Local search using cover_count for efficient updates.
        Supports 1-swap (change one residue) and optionally 2-swap (change two residues).
        """
        classes = [c[:] for c in classes]
        score = score
        
        # Initialize cover_count
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in classes:
            cover_count[r::m] += 1
            
        improved = True
        pass_count = 0
        max_passes = 30
        
        while improved and (time.time() - start_time) < time_limit and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            # 1-swap moves: change one residue
            for i in range(len(classes)):
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                best_r = r
                best_score = score
                
                # Try all possible residues for this modulus
                for new_r in range(m):
                    if new_r == r:
                        continue
                    
                    delta = fast_delta(cover_count, m, r, new_r)
                    new_score = score + delta
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_r = new_r

                if best_r != r:
                    # Apply the best change permanently
                    cover_count[r::m] -= 1
                    cover_count[best_r::m] += 1
                    classes[i] = [m, best_r]
                    score = best_score
                    improved = True
            
            # 2-swap moves: change two residues at once
            if allow_two_swap and (time.time() - start_time) < time_limit:
                for i in range(len(classes)):
                    if (time.time() - start_time) > time_limit:
                        break
                    m1, r1 = classes[i]
                    for j in range(i+1, len(classes)):
                        if (time.time() - start_time) > time_limit:
                            break
                        m2, r2 = classes[j]
                        
                        # Try a few random combinations of new residues
                        for _ in range(5):
                            new_r1 = random.randint(0, m1 - 1)
                            new_r2 = random.randint(0, m2 - 1)
                            if new_r1 == r1 and new_r2 == r2:
                                continue
                            
                            # Compute combined delta
                            delta1 = fast_delta(cover_count, m1, r1, new_r1)
                            # Temporarily update cover_count for m1 to compute delta2
                            cover_count[r1::m1] -= 1
                            cover_count[new_r1::m1] += 1
                            delta2 = fast_delta(cover_count, m2, r2, new_r2)
                            # Revert m1 change
                            cover_count[new_r1::m1] -= 1
                            cover_count[r1::m1] += 1
                            
                            delta = delta1 + delta2
                            new_score = score + delta
                            
                            if new_score < score:
                                # Apply both changes
                                cover_count[r1::m1] -= 1
                                cover_count[new_r1::m1] += 1
                                cover_count[r2::m2] -= 1
                                cover_count[new_r2::m2] += 1
                                classes[i] = [m1, new_r1]
                                classes[j] = [m2, new_r2]
                                score = new_score
                                improved = True
                                break  # Move to next i after finding an improvement

        return classes, score

    # Strategy 1: Greedy with all divisors in increasing order
    best_classes, best_score = greedy_init(divisors)
    
    # Run local search
    if (time.time() - start_time) < deadline - 10:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 5, allow_two_swap=True)

    # Strategy 2: Try different orderings
    random.seed(42)
    remaining = deadline - (time.time() - start_time)

    if remaining > 15:
        # Try greedy with shuffled order
        shuffled = divisors[:]
        random.shuffle(shuffled)
        classes2, score2 = greedy_init(shuffled)
        if score2 < best_score:
            best_classes, best_score = classes2, score2
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, best_score, deadline - 3, allow_two_swap=True)

    if (time.time() - start_time) < deadline - 15:
        # Try with larger moduli first
        divs_desc = sorted(divisors, reverse=True)
        classes3, score3 = greedy_init(divs_desc)
        if score3 < best_score:
            best_classes, best_score = classes3, score3
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, best_score, deadline - 3, allow_two_swap=True)

    # Strategy 3: Priority-based ordering (moduli that cover more unique residues first)
    if (time.time() - start_time) < deadline - 15:
        # Compute for each divisor m, the maximum number of residues it can cover
        # which is ceil(N/m). Sort by this in descending order.
        priority_order = sorted(divisors, key=lambda m: -N // m)
        classes4, score4 = greedy_init(priority_order)
        if score4 < best_score:
            best_classes, best_score = classes4, score4
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, best_score, deadline - 3, allow_two_swap=True)

    # Strategy 4: Simulated Annealing
    if (time.time() - start_time) < deadline - 10:
        current_classes = [c[:] for c in best_classes]
        current_score = best_score
        
        cover_count = build_cover_count(current_classes)
        
        temperature = max(1.0, best_score / 3.0)
        cooling_rate = 0.97
        min_temperature = 0.1
        
        sa_iterations = 0
        max_sa_iterations = 3000
        
        for iteration in range(max_sa_iterations):
            if (time.time() - start_time) > deadline - 5:
                break
            if temperature < min_temperature:
                break
                
            sa_iterations += 1
            
            # Perturb: change one residue
            idx = random.randint(0, len(current_classes) - 1)
            m, r = current_classes[idx]
            old_r = r
            new_r = random.randint(0, m - 1)
            
            if new_r == old_r:
                continue

            # Compute delta using fast method
            delta = fast_delta(cover_count, m, old_r, new_r)
            new_score = current_score + delta
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Metropolis criterion
                accept = random.random() < np.exp(-delta / max(temperature, 0.01))
            
            if accept:
                # Apply change
                cover_count[old_r::m] -= 1
                cover_count[new_r::m] += 1
                current_classes[idx] = [m, new_r]
                current_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in current_classes]
                    # Run local search from new best
                    if (time.time() - start_time) < deadline - 3:
                        best_classes, best_score = local_search(best_classes, best_score, deadline - 2, allow_two_swap=False)
                        # Re-sync cover_count with best_classes
                        cover_count = build_cover_count(best_classes)
                        current_classes = [c[:] for c in best_classes]
                        current_score = best_score
            else:
                # No change needed since we didn't modify cover_count
                pass

            # Cool down
            temperature *= cooling_rate

    # Strategy 5: Final aggressive local search with more passes
    if (time.time() - start_time) < deadline - 2:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 1, allow_two_swap=True)

    return best_classes
