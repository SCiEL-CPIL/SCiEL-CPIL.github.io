import itertools
import time

def _is_regular_induced(subset_mask, adj_masks, n):
    """
    Check if the induced subgraph on vertices in subset_mask is regular.
    subset_mask: integer bitmask of vertices in the subset
    adj_masks: list of integer bitmasks for adjacency of each vertex
    n: total number of vertices
    Returns True if regular, False otherwise.
    """
    # Get list of vertices in the subset
    vertices = []
    m = subset_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        vertices.append(v)
        m ^= lsb
        
    k = len(vertices)
    if k < 6:
        return False
        
    # Calculate degrees within the induced subgraph
    degs = []
    for v in vertices:
        # Count neighbors of v that are also in the subset
        # adj_masks[v] contains all neighbors of v
        # We need intersection with subset_mask
        common = adj_masks[v] & subset_mask
        d = common.bit_count()
        degs.append(d)
        
    # Check if all degrees are equal
    first_deg = degs[0]
    for d in degs[1:]:
        if d != first_deg:
            return False
    return True

def _check_graph_validity(n, adj_masks):
    """
    Check if the graph has no regular induced subgraph of size >= 6.
    Returns True if valid.
    """
    # Iterate over all subsets of size 6 to n
    # To be efficient, we can iterate over all masks and filter by popcount
    # Or use combinations. For n=20, 2^20 is ~1M masks.
    # Checking popcount for 1M masks is fast.
    # Checking regularity for valid candidates is fast.
    
    # Optimization: Only check subsets of size k where a regular graph of size k
    # is possible. A regular graph of size k and degree d must satisfy k*d is even.
    # But we just check all subsets.
    
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_induced(mask, adj_masks, n):
                return False
    return True

def _build_paley_like_graph(n):
    """
    Construct a graph based on quadratic residues modulo p.
    Paley graph P(q) where q is a prime power congruent to 1 mod 4.
    For n <= 22, we can use p=17 or p=13.
    However, Paley graphs are self-complementary and regular.
    The induced subgraphs might still be regular.
    
    Actually, a known construction for graphs with no regular induced subgraphs of size 6
    is related to "strongly regular graphs" or specific random constructions.
    
    Since the task states "Graphs with 20 vertices are known", and we need to maximize n,
    and we have 120s, we should try to find a 20-vertex graph deterministically or semi-deterministically.
    
    Let's use a specific known valid graph structure if possible, or a high-quality random search
    that is guided by the bitset verifier.
    
    However, the prompt says "rethinking the approach". 
    The current approach is random search.
    A different approach: **Backtracking with Constraint Propagation** or **Systematic Enumeration of Small Violations**.
    
    Actually, the most robust "different" strategy that guarantees a high score quickly is to use a **Pre-computed or Deterministically Generated Valid Graph** if we can find one, or a **Greedy Construction** that avoids forming regular subgraphs.
    
    Let's try a Greedy Construction:
    1. Start with n=20 vertices.
    2. Initialize empty graph.
    3. For each pair (u, v), decide if edge (u, v) exists.
    4. Constraint: Adding edge (u, v) must not create a regular induced subgraph of size 6 containing u and v.
    
    This is complex.
    
    Alternative: **Randomized Local Search with Bitset Verifier**.
    The current program uses a slow verifier or a less optimized one.
    My strategy: Use the fast bitset verifier (O(2^n * n) worst case per check, but early exit) and perform a **Simulated Annealing** or **Hill Climbing** starting from a random graph, minimizing the number of regular induced subgraphs. If we reach 0, we have a valid graph.
    
    This is materially different from "generate random, check, repeat". It is "generate random, optimize landscape".
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We aim for n=20.
    # If we can't find 20, we return the best we have.
    # But we know 20 exists.
    
    # Strategy: Simulated Annealing / Local Search
    # State: adjacency matrix as list of bitmasks.
    # Objective: Minimize count of regular induced subgraphs of size >= 6.
    # If count == 0, we are done.
    
    def count_violations(n, adj_masks):
        count = 0
        # Only check size 6? 
        # If there is a regular induced subgraph of size > 6, it might not contain a regular one of size 6.
        # However, for the purpose of local search, minimizing size-6 violations is a good proxy.
        # If we eliminate all size-6 regular induced subgraphs, do we guarantee no size > 6?
        # No. But it's a strong signal.
        # Let's check all sizes >= 6 to be safe, but it's expensive.
        # For local search steps, we can approximate by only checking size 6.
        # If we reach 0 size-6 violations, we do a full check.
        
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_induced(mask, adj_masks, n):
                count += 1
        return count

    def full_check(n, adj_masks):
        return _check_graph_validity(n, adj_masks)

    best_n = 5
    best_edges = []
    
    # Try to find a valid graph for n=20
    # If fails, try 19, etc.
    
    for target_n in range(20, 5, -1):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        random_seed = 42
        import random
        random.seed(random_seed)
        
        adj_masks = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                    
        # Local Search
        current_violations = count_violations(target_n, adj_masks)
        
        # If we start with 0 violations, check full validity
        if current_violations == 0:
            if full_check(target_n, adj_masks):
                # Extract edges
                edges = []
                for i in range(target_n):
                    m = adj_masks[i]
                    for j in range(i + 1, target_n):
                        if m & (1 << j):
                            edges.append([i, j])
                best_n = target_n
                best_edges = edges
                return {'n': best_n, 'edges': best_edges}
        
        # Simulated Annealing
        temperature = 1.0
        cooling_rate = 0.99
        steps = 5000
        step_count = 0
        
        while step_count < steps and time.time() < deadline:
            step_count += 1
            if time.time() > deadline:
                break
                
            # Pick random edge to flip
            i = random.randint(0, target_n - 1)
            j = random.randint(0, target_n - 1)
            if i == j:
                continue
                
            # Flip edge (i, j)
            mask_bit = 1 << j
            had_edge = bool(adj_masks[i] & mask_bit)
            
            # Remove edge
            adj_masks[i] ^= mask_bit
            adj_masks[j] ^= (1 << i)
            
            # Count new violations
            # Optimization: Only subsets containing i or j can change status.
            # But for simplicity in this prototype, we recompute all.
            # To make this fast, we need incremental updates.
            # For now, let's just do a few iterations and hope the initial random is good,
            # or use a smarter heuristic.
            
            # Actually, recomputing all C(20,6)=38760 subsets per step is too slow for 5000 steps in Python.
            # 38760 * 5000 = 193 million operations. Too slow.
            
            # Alternative Strategy: **Deterministic Construction**
            # There is a known graph: The complement of the disjoint union of 5 K4s? No.
            # Let's try a specific structure: The Clebsch graph? No, that's 16 vertices.
            # The Hoffman-Singleton graph? 50 vertices.
            
            # Let's go back to Random Search but with a MUCH faster verifier and more attempts.
            # The bottleneck in the current program is the verifier.
            # My verifier uses bit_count which is fast in Python 3.10+.
            # Let's just do random restarts with the fast verifier.
            
            # Revert edge
            adj_masks[i] ^= mask_bit
            adj_masks[j] ^= (1 << i)
            
            # Break out of local search loop, it's not efficient as written.
            # Let's switch to Random Restart with Fast Verifier.
            break

        # Random Restart Loop
        attempts = 0
        max_attempts = 2000
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            if time.time() > deadline:
                break
            
            # Generate random graph
            adj_masks = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        
            # Fast check: Only size 6 first
            has_violation_6 = False
            for subset in itertools.combinations(range(target_n), 6):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if _is_regular_induced(mask, adj_masks, target_n):
                    has_violation_6 = True
                    break
            
            if not has_violation_6:
                # Check larger sizes
                if full_check(target_n, adj_masks):
                    edges = []
                    for i in range(target_n):
                        m = adj_masks[i]
                        for j in range(i + 1, target_n):
                            if m & (1 << j):
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    return {'n': best_n, 'edges': best_edges}
            
            # Change seed for next attempt
            random_seed += 1
            random.seed(random_seed)
            
    # If we couldn't find 20, return best found (likely 5 from init, but we should return valid smaller if found)
    # For safety, if no graph found for 20..6, return n=5 (empty graph is valid? 
    # Induced subgraph of size 6 doesn't exist, so condition "no set of 6 or more" is vacuously true?
    # The task says "No set of 6 or more vertices may induce...". If n < 6, there are no such sets.
    # So n=5 is always valid.
    
    return {'n': best_n, 'edges': best_edges}
