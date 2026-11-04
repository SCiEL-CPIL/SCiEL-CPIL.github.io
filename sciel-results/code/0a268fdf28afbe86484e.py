"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import itertools

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the lower bound score for set U."""
    U = sorted(set(U))
    if len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute |U+U| and |U-U|
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _build_set_from_params(n_blocks, base, ratio, block_size):
    """Build a set U from interval parameters.
    
    n_blocks: number of intervals
    base: starting position of first block (must include 0)
    ratio: multiplicative ratio between gap sizes
    block_size: size of each interval (number of consecutive integers)
    """
    if n_blocks < 1:
        return [0]
    U = set()
    # First block starts at 0
    pos = 0
    gap = base if base > 0 else 1
    for i in range(n_blocks):
        for j in range(block_size):
            U.add(pos + j)
        if i < n_blocks - 1:
            pos += gap
            gap = int(gap * ratio + 0.5)
            if gap < 1:
                gap = 1
    # Ensure 0 is in the set
    U.add(0)
    return sorted(U)


def _build_geometric_set(n_elements, alpha):
    """Build a set where elements follow a geometric-like progression.
    
    U = {floor(alpha^i) for i in range(n_elements)}
    Ensures 0 is included.
    """
    if n_elements < 1:
        return [0]
    elements = [0]
    val = 1
    for i in range(1, n_elements):
        val = int(val * alpha + 0.5)
        if val <= elements[-1]:
            val = elements[-1] + 1
        elements.append(val)
    return elements


def _build_mixed_set(n_blocks, block_sizes, gaps):
    """Build a set from explicit block sizes and gaps."""
    U = set()
    pos = 0
    for i in range(n_blocks):
        for j in range(block_sizes[i]):
            U.add(pos + j)
        if i < n_blocks - 1:
            pos += gaps[i]
    U.add(0)
    return sorted(U)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Strategy 1: Geometric progression search
    # U = {0, floor(alpha^1), floor(alpha^2), ..., floor(alpha^(n-1))}
    for n_elements in range(3, 40):
        if time.time() > deadline:
            break
        # Search over alpha values
        for alpha_100 in range(101, 300):  # alpha from 1.01 to 3.00
            if time.time() > deadline:
                break
            alpha = alpha_100 / 100.0
            U = _build_geometric_set(n_elements, alpha)
            if len(U) < 2:
                continue
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U
    
    # Strategy 2: Interval-based with exponential gaps
    for n_blocks in range(2, 12):
        if time.time() > deadline:
            break
        for block_size in [1, 2, 3, 4, 5, 8, 12, 16]:
            if time.time() > deadline:
                break
            for ratio_100 in range(150, 400):  # ratio from 1.5 to 4.0
                if time.time() > deadline:
                    break
                ratio = ratio_100 / 100.0
                for initial_gap in [2, 3, 5, 8, 12, 20, 30, 50]:
                    if time.time() > deadline:
                        break
                    U = _build_set_from_params(n_blocks, initial_gap, ratio, block_size)
                    if len(U) < 2:
                        continue
                    score = _compute_score(U)
                    if score > best_score:
                        best_score = score
                        best_set = U
    
    # Strategy 3: Binary expansion subsets (from measured knowledge)
    # Try sets based on binary representations with selective bits
    if time.time() < deadline:
        for base in [2, 3, 4, 5, 7, 10]:
            if time.time() > deadline:
                break
            max_bits = min(12, int(math.log(5000, base)) + 2) if base > 1 else 12
            # Try subsets of positions in base representation
            for n_positions in range(2, min(8, max_bits + 1)):
                if time.time() > deadline:
                    break
                # Try geometric selection of bit positions
                for start_pos in range(0, max_bits - n_positions + 1):
                    if time.time() > deadline:
                        break
                    for pos_step in range(1, 4):
                        if time.time() > deadline:
                            break
                        positions = list(range(start_pos, start_pos + n_positions * pos_step, pos_step))[:n_positions]
                        if len(positions) < 2:
                            continue
                        # Build set from base^position
                        U = [0]
                        for p in positions:
                            val = base ** p
                            if val not in U:
                                U.append(val)
                        U = sorted(set(U))
                        if len(U) < 2:
                            continue
                        score = _compute_score(U)
                        if score > best_score:
                            best_score = score
                            best_set = U
    
    # Strategy 4: Local refinement around best set found
    if time.time() < deadline - 20 and best_set:
        # Try adding/removing elements near the best set
        best_set_copy = list(best_set)
        max_u = max(best_set_copy)
        # Try inserting elements
        for insert_val in range(0, max_u + 5):
            if time.time() > deadline:
                break
            if insert_val in best_set_copy:
                continue
            candidate = sorted(best_set_copy + [insert_val])
            score = _compute_score(candidate)
            if score > best_score:
                best_score = score
                best_set = candidate
    
    # Final validation
    if not best_set or 0 not in best_set:
        return [0, 1, 3]
    
    return sorted(set(best_set))

# EVOLVE-BLOCK-END
