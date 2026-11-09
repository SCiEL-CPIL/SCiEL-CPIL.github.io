<<<<<<< SEARCH
import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

def _check_pair(r1, m1, r2, m2):
    """Check if two residue classes are disjoint."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g

def _is_valid_solution(solution):
    """Check if a solution is valid: distinct moduli in [2,128], canonical residues, pairwise disjoint."""
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            r1, m1 = solution[i]
            r2, m2 = solution[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _greedy_construct(mod_order, deadline):
    """Greedy construction: for each modulus in order, pick the first valid residue."""
    solution = []
    used_moduli = set()
    
    for m in mod_order:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        found = False
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                if not _check_pair(r, m, r_prev, m_prev):
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                used_moduli.add(m)
                found = True
                break
    
    return solution

def _backtracking_construct(mod_order, deadline, max_depth=127):
    """Backtracking search to find a large valid set."""
    best_solution = []
    solution = []
    used_moduli = set()
    
    def backtrack(idx):
        nonlocal best_solution
        if time.time() > deadline:
            return
        if idx == len(mod_order):
            if len(solution) > len(best_solution):
                best_solution = [list(x) for x in solution]
            return
        if len(solution) + (len(mod_order) - idx) <= len(best_solution):
            return  # Prune: can't beat best
        
        m = mod_order[idx]
        # Try to add this modulus
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                if not _check_pair(r, m, r_prev, m_prev):
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                used_moduli.add(m)
                backtrack(idx + 1)
                solution.pop()
                used_moduli.discard(m)
                if time.time() > deadline:
                    return
        # Also try skipping this modulus
        backtrack(idx + 1)
    
    backtrack(0)
    return best_solution

def _local_search_improve(solution, deadline):
    """Try to improve the solution by reassigning residues or adding new moduli."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        
        # Try reassigning residues for each element to enable adding a new modulus
        for i in range(n):
            if time.time() > deadline:
                break
            r_i, m_i = best[i]
            # Try other residues for this modulus
            for r in range(m_i):
                if r == r_i:
                    continue
                # Check if new residue works with all others
                compatible = True
                for j in range(n):
                    if i == j:
                        continue
                    r_j, m_j = best[j]
                    if not _check_pair(r, m_i, r_j, m_j):
                        compatible = False
                        break
                if compatible:
                    new_solution = [list(x) for x in best]
                    new_solution[i] = [r, m_i]
                    
                    # Try to add a new modulus
                    used_moduli = set(m for _, m in new_solution)
                    for m_new in range(2, MAXIMUM + 1):
                        if m_new in used_moduli:
                            continue
                        for r_new in range(m_new):
                            compatible_new = True
                            for (r_prev, m_prev) in new_solution:
                                if not _check_pair(r_new, m_new, r_prev, m_prev):
                                    compatible_new = False
                                    break
                            if compatible_new:
                                new_solution.append([r_new, m_new])
                                if len(new_solution) > len(best):
                                    best = new_solution
                                    improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # Try adding a new modulus directly
        used_moduli = set(m for _, m in best)
        for m_new in range(2, MAXIMUM + 1):
            if time.time() > deadline:
                break
            if m_new in used_moduli:
                continue
            for r_new in range(m_new):
                compatible_new = True
                for (r_prev, m_prev) in best:
                    if not _check_pair(r_new, m_new, r_prev, m_prev):
                        compatible_new = False
                        break
                if compatible_new:
                    new_solution = [list(x) for x in best]
                    new_solution.append([r_new, m_new])
                    if len(new_solution) > len(best):
                        best = new_solution
                        improved = True
                    break
            if improved:
                break
    
    return best

def _smarter_greedy(moduli_list, deadline):
    """Greedy with dynamic ordering: at each step, pick the modulus that has the most remaining options."""
    solution = []
    remaining = list(moduli_list)
    
    while remaining and time.time() < deadline:
        best_m = None
        best_r = None
        best_score = -1
        
        for m in remaining:
            # Count how many valid residues exist
            valid_count = 0
            first_valid = None
            for r in range(m):
                compatible = True
                for (r_prev, m_prev) in solution:
                    if not _check_pair(r, m, r_prev, m_prev):
                        compatible = False
                        break
                if compatible:
                    valid_count += 1
                    if first_valid is None:
                        first_valid = r
            if valid_count > 0:
                # Prefer moduli with fewer valid options (more constrained first)
                score = valid_count
                if score < best_score or best_m is None:
                    best_score = score
                    best_m = m
                    best_r = first_valid
        
        if best_m is None:
            break
        
        solution.append([best_r, best_m])
        remaining.remove(best_m)
    
    return solution

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Phase 1: Multiple greedy strategies
    # Strategy 1: Simple greedy with various orderings
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: m.bit_count()),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
        sorted(all_moduli, key=lambda m: min(p for p in range(2, m+1) if m%p==0)),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        solution = _greedy_construct(mod_order, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Strategy 2: Smarter greedy with dynamic ordering
    if time.time() < deadline:
        solution = _smarter_greedy(all_moduli, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Phase 2: Local search improvement
    if best_solution and time.time() < deadline:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Phase 3: Randomized multi-start
    if time.time() < deadline:
        random.seed(42)
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            mod_order = all_moduli[:]
            random.shuffle(mod_order)
            
            solution = _greedy_construct(mod_order, deadline)
            if len(solution) > len(best_solution):
                best_solution = [list(x) for x in solution]
            
            if time.time() < deadline and len(solution) > 0:
                improved = _local_search_improve(solution, deadline)
                if len(improved) > len(best_solution):
                    best_solution = improved
    
    # Phase 4: Final local search
    if best_solution and time.time() < deadline:
        final = _local_search_improve(best_solution, deadline)
        if len(final) > len(best_solution):
            best_solution = final
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
=======
import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 3.0

def _check_pair(r1, m1, r2, m2):
    """Check if two residue classes are disjoint."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g

def _build_conflict_structure():
    """
    Build the conflict graph structure for branch-and-bound.
    
    Nodes: all valid (residue, modulus) pairs.
    Two nodes are compatible (can coexist) if their residue classes are disjoint.
    We want a maximum set of pairwise-compatible nodes with distinct moduli.
    
    Returns:
        nodes: list of (r, m) for all nodes
        compat: list of bitsets (as Python ints) where compat[i] has bit j set iff node i is compatible with node j
        mod_of_node: list of modulus for each node
        node_of_mod: dict mapping modulus -> list of node indices with that modulus
    """
    all_moduli = list(range(2, MAXIMUM + 1))
    nodes = []
    mod_of_node = []
    node_of_mod = {m: [] for m in all_moduli}
    
    for m in all_moduli:
        for r in range(m):
            idx = len(nodes)
            nodes.append((r, m))
            mod_of_node.append(m)
            node_of_mod[m].append(idx)
    
    n = len(nodes)
    # compat[i] is a bitmask: bit j is 1 if node i is compatible with node j
    compat = [0] * n
    
    for i in range(n):
        r_i, m_i = nodes[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            r_j, m_j = nodes[j]
            if m_i == m_j:
                continue  # Same modulus: cannot both be in solution
            g = gcd(m_i, m_j)
            if r_i % g != r_j % g:
                mask |= (1 << j)
        compat[i] = mask
    
    return nodes, compat, mod_of_node, node_of_mod

def _branch_and_bound(nodes, compat, mod_of_node, node_of_mod, deadline, initial_solution=None):
    """
    Branch-and-bound for maximum independent set with distinct modulus constraint.
    
    We search over moduli (not individual nodes). For each modulus, we choose at most one residue.
    
    Strategy:
    - Order moduli by some heuristic
    - At each step, try adding a modulus with a compatible residue, or skip it
    - Use upper bound pruning: current_size + number of remaining moduli <= best -> prune
    
    Uses bitsets for fast compatibility checking.
    """
    best_solution = []
    if initial_solution:
        best_solution = [list(x) for x in initial_solution]
    
    n_nodes = len(nodes)
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Precompute for each modulus, the list of node indices
    mod_to_nodes = {m: [] for m in all_moduli}
    for idx, (r, m) in enumerate(nodes):
        mod_to_nodes[m].append(idx)
    
    # For fast checking: given a set of selected nodes, which residues are valid for modulus m?
    # We'll compute this on the fly using bitsets.
    
    solution = []  # list of (r, m)
    selected_nodes = set()  # set of node indices
    
    # Order moduli: try primes first (they have gcd=1 with everything, so r=0 always works)
    def is_prime(n):
        if n < 2:
            return False
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                return False
        return True
    
    primes = [m for m in all_moduli if is_prime(m)]
    composites = [m for m in all_moduli if not is_prime(m)]
    
    # Try different orderings
    orderings = [
        primes + composites,
        composites + primes,
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
        sorted(all_moduli, key=lambda m: m.bit_count()),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        
        solution = []
        selected_nodes = set()
        
        def _compat_mask_for_mod(m, selected_nodes):
            """
            Return a bitmask of which residues for modulus m are compatible with all selected nodes.
            """
            if not selected_nodes:
                return (1 << m) - 1  # All residues valid
            
            # For each selected node, determine which residues of m are incompatible
            incompatible_mask = 0
            for node_idx in selected_nodes:
                r_sel, m_sel = nodes[node_idx]
                g = gcd(m, m_sel)
                # Residue r of m is incompatible with r_sel of m_sel if r % g == r_sel % g
                r_sel_mod_g = r_sel % g
                for r in range(m):
                    if r % g == r_sel_mod_g:
                        incompatible_mask |= (1 << r)
            
            return ((1 << m) - 1) & ~incompatible_mask
        
        def backtrack(idx):
            nonlocal best_solution
            if time.time() > deadline:
                return
            
            if idx == len(mod_order):
                if len(solution) > len(best_solution):
                    best_solution = [list(x) for x in solution]
                return
            
            # Upper bound: can we beat the best?
            if len(solution) + (len(mod_order) - idx) <= len(best_solution):
                return
            
            m = mod_order[idx]
            
            # Try each valid residue for this modulus
            valid_mask = _compat_mask_for_mod(m, selected_nodes)
            
            # Try residues in order of least constraining (prefer residues that leave more options)
            # For simplicity, try all valid residues
            residues = []
            for r in range(m):
                if valid_mask & (1 << r):
                    residues.append(r)
            
            for r in residues:
                # Find the node index for (r, m)
                node_idx = None
                for ni in mod_to_nodes[m]:
                    if nodes[ni][0] == r:
                        node_idx = ni
                        break
                
                if node_idx is None:
                    continue
                
                # Add this node
                solution.append([r, m])
                selected_nodes.add(node_idx)
                
                backtrack(idx + 1)
                
                solution.pop()
                selected_nodes.discard(node_idx)
                
                if time.time() > deadline:
                    return
            
            # Also try skipping this modulus
            backtrack(idx + 1)
        
        backtrack(0)
        
        if time.time() > deadline:
            break
    
    return best_solution

def _greedy_construct(mod_order, deadline):
    """Greedy construction: for each modulus in order, pick the first valid residue."""
    solution = []
    used_moduli = set()
    
    for m in mod_order:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                if not _check_pair(r, m, r_prev, m_prev):
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                used_moduli.add(m)
                break
    
    return solution

def _local_search_improve(solution, deadline):
    """Try to improve the solution by reassigning residues or adding new moduli."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        
        # Try reassigning residues for each element to enable adding a new modulus
        for i in range(n):
            if time.time() > deadline:
                break
            r_i, m_i = best[i]
            for r in range(m_i):
                if r == r_i:
                    continue
                compatible = True
                for j in range(n):
                    if i == j:
                        continue
                    r_j, m_j = best[j]
                    if not _check_pair(r, m_i, r_j, m_j):
                        compatible = False
                        break
                if compatible:
                    new_solution = [list(x) for x in best]
                    new_solution[i] = [r, m_i]
                    
                    used_moduli = set(m for _, m in new_solution)
                    for m_new in range(2, MAXIMUM + 1):
                        if m_new in used_moduli:
                            continue
                        for r_new in range(m_new):
                            compatible_new = True
                            for (r_prev, m_prev) in new_solution:
                                if not _check_pair(r_new, m_new, r_prev, m_prev):
                                    compatible_new = False
                                    break
                            if compatible_new:
                                new_solution.append([r_new, m_new])
                                if len(new_solution) > len(best):
                                    best = new_solution
                                    improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # Try adding a new modulus directly
        used_moduli = set(m for _, m in best)
        for m_new in range(2, MAXIMUM + 1):
            if time.time() > deadline:
                break
            if m_new in used_moduli:
                continue
            for r_new in range(m_new):
                compatible_new = True
                for (r_prev, m_prev) in best:
                    if not _check_pair(r_new, m_new, r_prev, m_prev):
                        compatible_new = False
                        break
                if compatible_new:
                    new_solution = [list(x) for x in best]
                    new_solution.append([r_new, m_new])
                    if len(new_solution) > len(best):
                        best = new_solution
                        improved = True
                    break
            if improved:
                break
    
    return best

def _smarter_greedy(moduli_list, deadline):
    """Greedy with dynamic ordering."""
    solution = []
    remaining = list(moduli_list)
    
    while remaining and time.time() < deadline:
        best_m = None
        best_r = None
        best_score = -1
        
        for m in remaining:
            valid_count = 0
            first_valid = None
            for r in range(m):
                compatible = True
                for (r_prev, m_prev) in solution:
                    if not _check_pair(r, m, r_prev, m_prev):
                        compatible = False
                        break
                if compatible:
                    valid_count += 1
                    if first_valid is None:
                        first_valid = r
            if valid_count > 0:
                score = valid_count
                if score < best_score or best_m is None:
                    best_score = score
                    best_m = m
                    best_r = first_valid
        
        if best_m is None:
            break
        
        solution.append([best_r, best_m])
        remaining.remove(best_m)
    
    return solution

def _bitset_branch_and_bound(deadline, initial_best):
    """
    Branch-and-bound using bitsets for fast compatibility checks.
    
    Key insight: For a given set of selected moduli with chosen residues, 
    the valid residues for a new modulus m can be computed quickly.
    
    We use the fact that for modulus m and selected (r_s, m_s), 
    residue r of m is invalid iff r % gcd(m, m_s) == r_s % gcd(m, m_s).
    """
    best = [list(x) for x in initial_best] if initial_best else []
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Precompute gcd table for all pairs of moduli
    gcd_table = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = gcd(m1, m2)
    
    def valid_residues_for(m, selected):
        """
        Return the set of valid residues for modulus m given selected list of (r, m).
        Uses bitset representation.
        """
        valid_mask = (1 << m) - 1  # All residues initially valid
        
        for (r_sel, m_sel) in selected:
            if m_sel == m:
                return 0  # Cannot use same modulus
            g = gcd_table[(m, m_sel)]
            r_sel_mod_g = r_sel % g
            # Invalidate all r where r % g == r_sel_mod_g
            for r in range(m):
                if r % g == r_sel_mod_g:
                    valid_mask &= ~(1 << r)
        
        return valid_mask
    
    def count_valid_residues(mask, m):
        """Count bits set in mask for modulus m."""
        count = 0
        for r in range(m):
            if mask & (1 << r):
                count += 1
        return count
    
    # Order moduli for search
    def is_prime(n):
        if n < 2:
            return False
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                return False
        return True
    
    primes = [m for m in all_moduli if is_prime(m)]
    composites = [m for m in all_moduli if not is_prime(m)]
    
    # Try multiple orderings
    orderings = [
        primes + composites,
        composites + primes,
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
        sorted(all_moduli, key=lambda m: m.bit_count()),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
        sorted(all_moduli, key=lambda m: len([p for p in range(2, m) if gcd(m, p) > 1])),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        
        selected = []  # list of (r, m)
        
        def backtrack(idx):
            nonlocal best
            if time.time() > deadline:
                return
            
            # Upper bound pruning
            if len(selected) + (len(mod_order) - idx) <= len(best):
                return
            
            if idx == len(mod_order):
                if len(selected) > len(best):
                    best = [list(x) for x in selected]
                return
            
            m = mod_order[idx]
            
            # Compute valid residues
            valid_mask = valid_residues_for(m, selected)
            
            # Try each valid residue
            for r in range(m):
                if valid_mask & (1 << r):
                    selected.append((r, m))
                    backtrack(idx + 1)
                    selected.pop()
                    if time.time() > deadline:
                        return
            
            # Skip this modulus
            backtrack(idx + 1)
        
        backtrack(0)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Phase 1: Quick greedy to get a starting solution
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: m.bit_count()),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        solution = _greedy_construct(mod_order, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Smarter greedy
    if time.time() < deadline:
        solution = _smarter_greedy(all_moduli, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Phase 2: Local search
    if best_solution and time.time() < deadline:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Phase 3: Branch and bound with bitsets
    if time.time() < deadline:
        bnb_solution = _bitset_branch_and_bound(deadline, best_solution)
        if len(bnb_solution) > len(best_solution):
            best_solution = bnb_solution
    
    # Phase 4: Final local search polish
    if best_solution and time.time() < deadline:
        final = _local_search_improve(best_solution, deadline)
        if len(final) > len(best_solution):
            best_solution = final
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
>>>>>>> REPLACE
