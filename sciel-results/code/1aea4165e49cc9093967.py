# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free subset of Z/205Z.
    Uses CRT structure (205 = 5*41) for initialization and bitmask-based
    local search for optimization.
    """
    m = 205
    deadline = time.monotonic() + 600.0  # safety margin before 630s kill

    # --- Precomputation ---
    forbidden = set()
    for x in range(1, m):
        f = (x * x) % m
        if f != 0:
            forbidden.add(f)
    forbidden.discard(0)

    # Verify F is closed under negation
    # Build conflict bitmask: conflict_mask[i] is a bitmask of all j that conflict with i
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict_mask[i] = mask

    # For fast checking: can_add(candidate, selected_mask) returns True if no conflict
    # selected_mask is a Python int bitmask

    def can_add(cand, selected_mask):
        """Check if cand can be added to the set represented by selected_mask."""
        # cand conflicts with any selected element if (1<<cand) & conflict_mask[i] for any i in selected
        # Equivalently: for each i in selected, if cand is in conflict_mask[i]
        # Faster: check if selected_mask & conflict_mask[cand] == 0
        # Wait: conflict_mask[cand] has bits set for all j that conflict with cand
        # So if any bit in selected_mask is also in conflict_mask[cand], there's a conflict
        return (selected_mask & conflict_mask[cand]) == 0

    def count_bits(mask):
        return bin(mask).count('1')

    # --- CRT-based initialization ---
    # Z/205Z ≅ Z/5Z × Z/41Z
    # Quadratic residues mod 5 (nonzero): {1, 4}
    # Quadratic residues mod 41 (nonzero): 
    qr41 = set()
    for x in range(1, 41):
        qr41.add((x * x) % 41)
    qr41.discard(0)
    # qr41 has 20 elements

    # A pair (a,b) in Z/5 × Z/41 is forbidden as a difference if
    # (a mod 5, b mod 41) is in the set of forbidden difference pairs
    # Forbidden differences mod 205 correspond to (qr5_elem, qr41_elem) where
    # qr5_elem in {1,4} and qr41_elem in qr41
    # So a difference (d5, d41) is forbidden if d5 in {1,4} AND d41 in qr41
    # For a set S to be valid: for any two elements (s5, s41), (t5, t41) in S,
    # (s5-t5 mod 5, s41-t41 mod 41) must NOT be in the forbidden set
    # i.e., NOT (s5-t5 mod 5 in {1,4} AND s41-t41 mod 41 in qr41)
    # Equivalently: for any pair, either s5-t5 mod 5 in {0,2,3} OR s41-t41 mod 41 not in qr41

    # Strategy: pick a small set of residues mod 5, and for each, pick residues mod 41
    # such that the difference condition holds

    # Let's try: use only elements where the mod-5 component is from a small set
    # If we use only mod-5 = 0: differences mod 5 are always 0, which is not in {1,4}, so valid!
    # That gives us 41 elements (all x with x mod 5 = 0)
    # Similarly mod-5 = 1: differences can be 0,1,4 which includes 1,4 -> need to check mod 41
    # Let's start with the easy case: all elements congruent to 0 mod 5
    # That's 41 elements: {0, 5, 10, 15, ..., 200}
    # Check: difference between any two is multiple of 5. Is a multiple of 5 a square mod 205?
    # 5k mod 205: for this to be a square, we need 5k = x^2 mod 205
    # x^2 mod 5 = 0, so x must be divisible by 5. Let x = 5y, then x^2 = 25y^2
    # 25y^2 mod 205 = 25y^2 (since 25y^2 < 205 for small y, but in general mod 205)
    # 25y^2 mod 205: 205 = 5*41, so 25y^2 mod 5 = 0, 25y^2 mod 41 = 25y^2 mod 41
    # For 25y^2 to be a multiple of 5 mod 205, it's always true.
    # For it to equal 5k, we need 25y^2 ≡ 5k (mod 205), i.e., 5y^2 ≡ k (mod 41)
    # So 5k is a square mod 205 iff k = 5y^2 mod 41 for some y, i.e., k/5 is a square mod 41 (when 5|k)
    # Actually this is getting complex. Let me just verify computationally.

    # Simple approach: start with greedy + local search using bitmasks

    def randomized_greedy(rng):
        order = list(range(m))
        rng.shuffle(order)
        selected_mask = 0
        for cand in order:
            if can_add(cand, selected_mask):
                selected_mask |= (1 << cand)
        return selected_mask

    def local_search(initial_mask, rng, max_iters=100000):
        """Local search with add, remove, swap moves."""
        current_mask = initial_mask
        cur_size = count_bits(current_mask)
        best_mask = initial_mask
        best_size = cur_size
        no_improve = 0
        patience = 20000

        for it in range(max_iters):
            if time.monotonic() > deadline:
                break
            if no_improve > patience:
                break

            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a random element
                cand = rng.randint(0, m - 1)
                if not (current_mask & (1 << cand)):
                    if can_add(cand, current_mask):
                        current_mask |= (1 << cand)
                        if count_bits(current_mask) > best_size:
                            best_mask = current_mask
                            best_size = count_bits(current_mask)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            elif move_type < 0.6:
                # Try to remove a random element (to enable future adds)
                if current_mask:
                    # Pick a random set bit
                    bits = []
                    tmp = current_mask
                    while tmp:
                        lsb = tmp & (-tmp)
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        tmp ^= lsb
                    if bits:
                        to_remove = rng.choice(bits)
                        current_mask &= ~(1 << to_remove)
                        no_improve += 1

            else:
                # Swap: remove one, try to add another
                if current_mask:
                    bits = []
                    tmp = current_mask
                    while tmp:
                        lsb = tmp & (-tmp)
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        tmp ^= lsb
                    if bits:
                        to_remove = rng.choice(bits)
                        current_mask &= ~(1 << to_remove)
                        # Try several candidates to add
                        improved = False
                        for _ in range(50):
                            cand = rng.randint(0, m - 1)
                            if not (current_mask & (1 << cand)):
                                if can_add(cand, current_mask):
                                    current_mask |= (1 << cand)
                                    if count_bits(current_mask) > best_size:
                                        best_mask = current_mask
                                        best_size = count_bits(current_mask)
                                        no_improve = 0
                                    improved = True
                                    break
                        if not improved:
                            # Undo removal
                            current_mask |= (1 << to_remove)
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

        return best_mask

    def simulated_annealing(initial_mask, rng, steps=200000):
        """Simulated annealing on the bitmask representation."""
        current_mask = initial_mask
        cur_size = count_bits(current_mask)
        best_mask = initial_mask
        best_size = cur_size
        T = 2.0
        cooling = 0.99995

        for step in range(steps):
            if time.monotonic() > deadline:
                break

            move_type = rng.random()
            
            if move_type < 0.35:
                # Add
                cand = rng.randint(0, m - 1)
                if not (current_mask & (1 << cand)):
                    if can_add(cand, current_mask):
                        current_mask |= (1 << cand)
                        if count_bits(current_mask) > best_size:
                            best_mask = current_mask
                            best_size = count_bits(current_mask)
                    else:
                        # Accept with probability exp(-1/T) (penalty for infeasibility not applicable)
                        pass
            elif move_type < 0.55:
                # Remove
                if current_mask:
                    bits = []
                    tmp = current_mask
                    while tmp:
                        lsb = tmp & (-tmp)
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        tmp ^= lsb
                    if bits:
                        to_remove = rng.choice(bits)
                        current_mask &= ~(1 << to_remove)
            else:
                # Swap
                if current_mask:
                    bits = []
                    tmp = current_mask
                    while tmp:
                        lsb = tmp & (-tmp)
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        tmp ^= lsb
                    if bits:
                        to_remove = rng.choice(bits)
                        current_mask &= ~(1 << to_remove)
                        cand = rng.randint(0, m - 1)
                        if not (current_mask & (1 << cand)):
                            if can_add(cand, current_mask):
                                current_mask |= (1 << cand)
                                if count_bits(current_mask) > best_size:
                                    best_mask = current_mask
                                    best_size = count_bits(current_mask)
                            else:
                                current_mask |= (1 << to_remove)
                        else:
                            current_mask |= (1 << to_remove)

            T *= cooling

        return best_mask

    def bitmask_to_set(mask):
        result = []
        for i in range(m):
            if mask & (1 << i):
                result.append(i)
        return result

    # --- Main search ---
    best_mask = 0
    best_size = 0

    rng = random.Random(42)

    # Phase 1: Many randomized greedy restarts
    greedy_count = 0
    while time.monotonic() < deadline - 60:  # reserve 60s for polish
        sol_mask = randomized_greedy(rng)
        greedy_count += 1
        sol_size = count_bits(sol_mask)
        if sol_size > best_size:
            best_mask = sol_mask
            best_size = sol_size
        # Periodically run local search on best
        if greedy_count % 3 == 0:
            improved = local_search(best_mask, rng, max_iters=50000)
            imp_size = count_bits(improved)
            if imp_size > best_size:
                best_mask = improved
                best_size = imp_size

    # Phase 2: Simulated annealing from best
    if time.monotonic() < deadline - 10:
        improved = simulated_annealing(best_mask, rng, steps=300000)
        imp_size = count_bits(improved)
        if imp_size > best_size:
            best_mask = improved
            best_size = imp_size

    # Phase 3: Final intensive local search
    if time.monotonic() < deadline - 3:
        improved = local_search(best_mask, rng, max_iters=200000)
        imp_size = count_bits(improved)
        if imp_size > best_size:
            best_mask = improved
            best_size = imp_size

    # Convert to sorted array
    result_list = bitmask_to_set(best_mask)
    if not result_list:
        result_list = [0]

    result = np.array(sorted(result_list), dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
