import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s

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
    n_div = len(divisors)

    # Precompute position arrays for each divisor and residue
    pos_arrays = {}
    for m in divisors:
        pos_arrays[m] = []
        for r in range(m):
            positions = np.arange(r, N, m, dtype=np.int32)
            pos_arrays[m].append(positions)

    def compute_score(classes):
        """Compute number of uncovered residues using numpy."""
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            positions = pos_arrays[m][r]
            covered[positions] = 1
        return int(np.sum(1 - covered))

    def get_covered(classes):
        """Get the covered boolean array."""
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            positions = pos_arrays[m][r]
            covered[positions] = 1
        return covered

    def greedy_init(moduli_list):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        covered = np.zeros(N, dtype=np.uint8)
        classes = []
        for m in moduli_list:
            best_r = 0
            best_count = -1
            for r in range(m):
                positions = pos_arrays[m][r]
                cnt = int(np.sum(1 - covered[positions]))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            positions = pos_arrays[m][best_r]
            covered[positions] = 1
        uncovered = int(np.sum(1 - covered))
        return classes, uncovered, covered

    def local_search_fast(classes, score, time_limit):
        """
        Local search using coverage count array.
        coverage_count[pos] = number of classes covering position pos.
        """
        classes = [c[:] for c in classes]
        
        # Build coverage count array
        cov_count = np.zeros(N, dtype=np.int32)
        for m, r in classes:
            positions = pos_arrays[m][r]
            cov_count[positions] += 1
        
        uncovered = int(np.sum(cov_count == 0))
        if uncovered != score:
            score = uncovered  # Use actual score

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(len(classes)):
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                best_r = r
                best_score = score

                old_positions = pos_arrays[m][r]

                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_positions = pos_arrays[m][new_r]
                    
                    # Compute delta in uncovered count
                    # Positions that become uncovered: in old_positions where cov_count == 1 (only this class)
                    # Positions that become covered: in new_positions where cov_count == 0
                    loss = int(np.sum((cov_count[old_positions] == 1)))
                    gain = int(np.sum((cov_count[new_positions] == 0)))
                    
                    new_score = score - gain + loss
                    if new_score < best_score:
                        best_score = new_score
                        best_r = new_r

                if best_r != r:
                    # Update coverage count
                    old_positions = pos_arrays[m][r]
                    new_positions = pos_arrays[m][best_r]
                    cov_count[old_positions] -= 1
                    cov_count[new_positions] += 1
                    classes[i] = [m, best_r]
                    score = best_score
                    improved = True

        return classes, score

    # Strategy 1: Greedy with all divisors in increasing order
    best_classes, best_score, _ = greedy_init(divisors)

    # Run local search
    if (time.time() - start_time) < deadline - 10:
        best_classes, best_score = local_search_fast(best_classes, best_score, deadline - 5)

    # Strategy 2: Try different orderings
    random.seed(42)
    remaining = deadline - (time.time() - start_time)

    if remaining > 15:
        shuffled = divisors[:]
        random.shuffle(shuffled)
        classes2, score2, _ = greedy_init(shuffled)
        if score2 < best_score:
            best_classes, best_score = classes2, score2
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search_fast(best_classes, best_score, deadline - 3)

    if (time.time() - start_time) < deadline - 15:
        divs_desc = sorted(divisors, reverse=True)
        classes3, score3, _ = greedy_init(divs_desc)
        if score3 < best_score:
            best_classes, best_score = classes3, score3
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search_fast(best_classes, best_score, deadline - 3)

    # Strategy 3: Simulated annealing
    if (time.time() - start_time) < deadline - 10:
        current_classes = [c[:] for c in best_classes]
        current_score = best_score
        temperature = max(1, best_score // 5)

        # Build cov_count for current
        cov_count = np.zeros(N, dtype=np.int32)
        for m, r in current_classes:
            positions = pos_arrays[m][r]
            cov_count[positions] += 1

        for iteration in range(500):
            if (time.time() - start_time) > deadline - 5:
                break

            idx = random.randint(0, len(current_classes) - 1)
            m, r = current_classes[idx]
            new_r = random.randint(0, m - 1)
            if new_r == r:
                continue

            old_positions = pos_arrays[m][r]
            new_positions = pos_arrays[m][new_r]

            loss = int(np.sum((cov_count[old_positions] == 1)))
            gain = int(np.sum((cov_count[new_positions] == 0)))
            delta = loss - gain  # change in uncovered count

            if delta < 0 or random.random() < (1.0 / (1.0 + delta / max(1, temperature))):
                # Accept
                cov_count[old_positions] -= 1
                cov_count[new_positions] += 1
                current_classes[idx] = [m, new_r]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in current_classes]
                    # Run local search from new best
                    if (time.time() - start_time) < deadline - 3:
                        best_classes, best_score = local_search_fast(best_classes, best_score, deadline - 2)
                        # Rebuild cov_count for current
                        cov_count = np.zeros(N, dtype=np.int32)
                        for m2, r2 in current_classes:
                            positions = pos_arrays[m2][r2]
                            cov_count[positions] += 1
            else:
                # Reject, no change needed

            temperature = max(1, temperature * 0.98)

    # Final local search if time permits
    if (time.time() - start_time) < deadline - 2:
        best_classes, best_score = local_search_fast(best_classes, best_score, deadline - 1)

    return best_classes
