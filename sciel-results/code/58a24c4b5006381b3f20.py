"""Combined: Correct gcd>1 clique constraint + greedy initialization + swap-enabled local search
to maximize disjoint residue classes with distinct moduli 2..128."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START = None
MAX_MOD = 128


def _prime_factors(m):
    """Distinct prime factors of m."""
    facs = set()
    d = 2
    while d * d <= m:
        if m % d == 0:
            facs.add(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        facs.add(m)
    return facs


def construct():
    global _START
    if _START is None:
        _START = time.time()

    deadline = _START + 85.0  # Safety margin before 90s kill

    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute omega for ordering heuristics
    def omega(m):
        count = 0
        d = 2
        while d * d <= m:
            if m % d == 0:
                count += 1
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            count += 1
        return count

    omega_cache = {m: omega(m) for m in moduli}

    # Precompute orderings for greedy initialization
    orderings = []
    orderings.append(sorted(moduli, key=lambda m: (-omega_cache[m], m)))  # Most constrained first
    orderings.append(sorted(moduli, key=lambda m: (omega_cache[m], m)))   # Least constrained first
    orderings.append(sorted(moduli))                                       # Ascending
    orderings.append(sorted(moduli, reverse=True))                         # Descending
    orderings.append(sorted(moduli, key=lambda m: (-omega_cache[m], -m)))  # Most constrained, largest first

    def _find_residue(m_idx, selected_pairs, selected_mod_indices):
        """
        Find a residue r in [0, moduli[m_idx]) disjoint from all selected classes.
        Returns r or None if no such residue exists.
        """
        m = moduli[m_idx]
        constraints = []
        for k, (r_i, m_i) in enumerate(selected_pairs):
            m_i_idx = selected_mod_indices[k]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))

        if not constraints:
            return 0

        # Efficiently check residues
        # For each residue r in [0, m), check if it conflicts with any constraint
        # Conflict if (r - r_i) % g == 0
        # We can mark forbidden residues or just check on the fly.
        # Since m <= 128, on-the-fly is fast enough.
        for r in range(m):
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def _can_add(m_idx, selected_mod_indices):
        """Check if modulus m_idx can be added based on gcd compatibility."""
        for m_sel_idx in selected_mod_indices:
            if gcd_table[m_idx][m_sel_idx] == 1:
                return False
        return True

    def greedy_pass(order):
        """One greedy pass given an ordering of moduli. Returns list of (r, m) pairs."""
        selected_pairs = []
        selected_mod_indices = []
        used = set()

        for m in order:
            if m in used:
                continue
            m_idx = mod_index[m]

            if not _can_add(m_idx, selected_mod_indices):
                continue

            r = _find_residue(m_idx, selected_pairs, selected_mod_indices)
            if r is not None:
                selected_pairs.append((r, m))
                selected_mod_indices.append(m_idx)
                used.add(m)

        return selected_pairs

    best_pairs = []
    best_count = 0

    # 1. Run deterministic orderings
    for order in orderings:
        if time.time() > deadline:
            break
        pairs = greedy_pass(order)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # 2. Randomized restarts for better initial solutions
    random.seed(42)
    num_restarts = 2000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        shuffled = moduli[:]
        random.shuffle(shuffled)
        pairs = greedy_pass(shuffled)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # 3. Local Search with Swaps
    # Strategy: 
    # - Try to add unused moduli.
    # - If stuck, try swapping a selected modulus with an unused one to see if it allows more additions.
    # - Or, simply try swapping residues of selected moduli to allow a new addition (though residues are already optimized for disjointness, 
    #   changing one might break another, so we must re-verify).
    # Actually, the primary bottleneck is the *set* of moduli. The residues are secondary.
    # A swap of modulus m_old with m_new might allow adding m_new and keeping others, 
    # or might allow adding a third modulus m_third that was previously blocked by m_old or m_new's specific gcd interactions.
    
    # Let's implement a local search that:
    # 1. Attempts to add any unused modulus.
    # 2. If no addition possible, attempts to swap a selected modulus with an unused one.
    #    - Remove m_old, add m_new.
    #    - Then try to add all other unused moduli again.
    #    - If the new size is larger, keep it.

    def _local_search(pairs):
        """Perform local search on the given pairs. Returns improved pairs."""
        current_pairs = pairs[:]
        current_count = len(current_pairs)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            used = set(m for _, m in current_pairs)
            current_mod_indices = [mod_index[m] for _, m in current_pairs]
            
            # Phase 1: Try to add any unused modulus
            added_any = False
            for m in moduli:
                if time.time() > deadline - 2.0:
                    break
                if m in used:
                    continue
                m_idx = mod_index[m]
                if not _can_add(m_idx, current_mod_indices):
                    continue
                r = _find_residue(m_idx, current_pairs, current_mod_indices)
                if r is not None:
                    current_pairs.append((r, m))
                    current_mod_indices.append(m_idx)
                    used.add(m)
                    added_any = True
                    improved = True
                    # Re-evaluate after addition? 
                    # Greedy addition might block others, but we loop through all.
                    # It's better to restart the "try to add" loop or just continue.
                    # Continuing is fine, but we should update current_mod_indices.
            
            if added_any:
                continue # Restart the while loop to check for more additions
            
            # Phase 2: Try swaps if no addition was possible
            # Pick a random selected modulus to remove, try to replace it with an unused one.
            # To be efficient, we can try a few random swaps.
            
            # We need to be careful: removing a modulus might allow adding several others.
            # So: Remove m_old. Try to add all unused moduli greedily.
            # If the resulting set is larger than current_count, accept.
            
            # To avoid O(N^2) complexity in the inner loop, we limit the number of swap attempts.
            max_swap_attempts = 20
            for _ in range(max_swap_attempts):
                if time.time() > deadline - 2.0:
                    break
                if not current_pairs:
                    break
                
                # Pick a random index to remove
                idx_to_remove = random.randint(0, len(current_pairs) - 1)
                r_old, m_old = current_pairs[idx_to_remove]
                
                # Find an unused modulus to potentially swap in
                # Ideally one that has better connectivity or just random.
                unused_moduli = [m for m in moduli if m not in used]
                if not unused_moduli:
                    break
                
                m_new = random.choice(unused_moduli)
                
                # Construct a new set: current_pairs without m_old, plus m_new
                new_pairs = [p for i, p in enumerate(current_pairs) if i != idx_to_remove]
                new_used = set(m for _, m in new_pairs)
                new_mod_indices = [mod_index[m] for _, m in new_pairs]
                
                # Try to add m_new
                m_new_idx = mod_index[m_new]
                if _can_add(m_new_idx, new_mod_indices):
                    r_new = _find_residue(m_new_idx, new_pairs, new_mod_indices)
                    if r_new is not None:
                        new_pairs.append((r_new, m_new))
                        new_used.add(m_new)
                        new_mod_indices.append(m_new_idx)
                
                # Now, try to add ANY other unused moduli to this new set
                # This is the key: the swap might open up space for more moduli.
                # We do a greedy pass on the remaining unused moduli.
                for m_add in moduli:
                    if time.time() > deadline - 2.0:
                        break
                    if m_add in new_used:
                        continue
                    m_add_idx = mod_index[m_add]
                    if not _can_add(m_add_idx, new_mod_indices):
                        continue
                    r_add = _find_residue(m_add_idx, new_pairs, new_mod_indices)
                    if r_add is not None:
                        new_pairs.append((r_add, m_add))
                        new_used.add(m_add)
                        new_mod_indices.append(m_add_idx)
                        
                if len(new_pairs) > current_count:
                    current_pairs = new_pairs
                    current_count = len(new_pairs)
                    improved = True
                    break # Restart the while loop with the new state
        
        return current_pairs

    # Apply local search to the best solution found so far
    if best_pairs:
        improved_pairs = _local_search(best_pairs)
        if len(improved_pairs) > best_count:
            best_pairs = improved_pairs
            best_count = len(best_pairs)

    # Additional randomized local search passes starting from subsets
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            if not best_pairs:
                break
            # Start from a random subset of the best solution
            subset_size = max(1, len(best_pairs) // 2)
            subset = random.sample(best_pairs, min(subset_size, len(best_pairs)))
            current_pairs = subset[:]
            
            # Run local search on this subset
            improved_pairs = _local_search(current_pairs)
            if len(improved_pairs) > best_count:
                best_count = len(improved_pairs)
                best_pairs = improved_pairs[:]

    # Safety fallback
    if not best_pairs:
        best_pairs = [(1, 2)]

    # Enforce canonical residues and distinct moduli
    seen = set()
    out = []
    for r, m in best_pairs:
        if m in seen:
            continue
        if not (0 <= r < m):
            continue
        seen.add(m)
        out.append([r, m])

    if not out:
        out = [[1, 2]]

    return out
# EVOLVE-BLOCK-END
