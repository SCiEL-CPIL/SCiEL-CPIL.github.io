import time
import random
import itertools

def is_sidon(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    for k in range(n, 0, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def has_sidon_subset(A, k):
    """
    Check if A has ANY Sidon subset of size k.
    Returns True if found, False otherwise.
    """
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def count_sidon_subsets(A, k):
    """
    Count the number of Sidon subsets of size k in A.
    Lower count means harder to find a Sidon subset.
    """
    count = 0
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Initial candidate: contiguous block
    current_A = list(range(12))
    best_A = list(current_A)
    best_val = max_sidon_size(best_A)  # expected 4

    # Smooth surrogate: count of Sidon subsets of size best_val
    # If best_val is 4, minimize count of size-4 Sidon subsets
    target_k = best_val
    current_count = count_sidon_subsets(current_A, target_k)
    best_count = current_count

    iterations = 0
    no_improve = 0
    best_hard_val = best_val

    while time.time() < deadline:
        iterations += 1

        # Create perturbed copy
        A_new = list(current_A)
        idx = random.randint(0, 11)

        # Biased perturbation: stay in dense region to encourage collisions
        if random.random() < 0.7:
            # Small step
            step = random.randint(-3, 3)
            new_val = A_new[idx] + step
        else:
            # Random in window around current set
            min_val = min(current_A)
            max_val = max(current_A)
            window = max(5, (max_val - min_val) // 2)
            new_val = random.randint(min_val - window, max_val + window)

        # Ensure validity
        if new_val < 0 or new_val > 128:
            new_val = random.randint(0, 128)

        if new_val in A_new:
            continue

        A_new[idx] = new_val

        # Fast check: does A_new still have a Sidon subset of size best_hard_val?
        # If not, the hard max has dropped — big win
        if best_hard_val >= 2:
            if not has_sidon_subset(A_new, best_hard_val):
                # Hard max dropped below best_hard_val
                new_hard = max_sidon_size(A_new)
                if new_hard < best_hard_val:
                    best_hard_val = new_hard
                    best_val = new_hard
                    best_A = A_new
                    current_A = A_new
                    target_k = best_val
                    current_count = count_sidon_subsets(current_A, target_k)
                    best_count = current_count
                    no_improve = 0
                    if best_hard_val <= 3:
                        # Found a 3 or lower — excellent
                        break
                    continue

        # Smooth surrogate: count of Sidon subsets of size target_k
        new_count = count_sidon_subsets(A_new, target_k)

        if new_count < current_count:
            # Improved smooth metric
            current_count = new_count
            current_A = A_new
            no_improve = 0

            # Periodically check hard max to update best_A
            if iterations % 50 == 0 or new_count == 0:
                hard_val = max_sidon_size(A_new)
                if hard_val < best_hard_val:
                    best_hard_val = hard_val
                    best_val = hard_val
                    best_A = A_new
                    target_k = best_val
                    current_count = count_sidon_subsets(current_A, target_k)
                    best_count = current_count
                    if best_hard_val <= 3:
                        break
        elif new_count == current_count:
            # Plateau: accept with probability to explore
            if random.random() < 0.2:
                current_A = A_new
        else:
            # Worse smooth metric
            no_improve += 1

        # Periodic reset if stuck
        if no_improve > 2000:
            current_A = list(best_A)
            current_count = count_sidon_subsets(current_A, target_k)
            no_improve = 0

        # Final safety check
        if time.time() > deadline - 2.0:
            break

    # Final verification and return
    final_val = max_sidon_size(best_A)
    return best_A
