"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _verify_55_free(adj, n, cliquesize=5, indsize=5):
    """Check that graph has no clique of size cliquesize and no independent set of size indsize.
    adj is a list of sets (or bitmasks). Here we use a list of Python ints as bitmasks for speed."""
    # We'll use bitmasks for adjacency. adj[i] is an int where bit j is set if i~j.
    # Check for K5: brute force is too slow for n>40. We use a recursive backtracking with pruning.
    # But for n=42, K5 check is fast enough with bit operations.
    
    def has_k_clique(adj, n, k):
        # Returns True if there is a clique of size k
        # Use bitmasks
        # Recursive: pick a vertex, then find clique of size k-1 in its neighbors
        # Optimization: only consider vertices with index > last to avoid duplicates? 
        # Actually, standard clique finding:
        
        # We can use a simple DFS. For n=42, k=5, this is very fast.
        # State: current clique size, available candidates (bitmask)
        # To ensure we don't miss, we just need to find ANY clique.
        
        # Let's use a helper that checks if a bitmask contains a clique of size k
        def _has_clique(candidates, k):
            if k == 1:
                return candidates != 0
            if k == 0:
                return True
            # Pick the lowest set bit
            # If candidates is 0, no clique
            if candidates == 0:
                return False
            # Get lowest set bit
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            # Option 1: include v
            # Neighbors of v in candidates
            nbrs = adj[v] & candidates
            if _has_clique(nbrs, k - 1):
                return True
            # Option 2: exclude v
            candidates_without_v = candidates & ~lsb
            if _has_clique(candidates_without_v, k):
                return True
            return False
        
        full_mask = (1 << n) - 1
        return _has_clique(full_mask, k)

    def has_ind_set(adj, n, k):
        # Independent set of size k is a clique of size k in the complement
        # Complement adjacency: comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        full_mask = (1 << n) - 1
        comp_adj = [full_mask ^ (1 << i) ^ adj[i] for i in range(n)]
        return has_k_clique(comp_adj, n, k)

    if has_k_clique(adj, n, cliquesize):
        return False
    if has_ind_set(adj, n, indsize):
        return False
    return True

def _paley_37():
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

def _exoo_42():
    # Exoo's 42-vertex (5,5)-free graph.
    # This is the Paley graph of order 42? No, Paley graphs exist for q = p^k = 1 mod 4.
    # 42 is not of the form p^k.
    # The 42-vertex graph is a known construction.
    # It is often described as a "Paley-type" or via a specific edge list.
    # Let's try to reconstruct it or find a known edge list.
    # Actually, the best known lower bound R(5,5) >= 43 comes from a 42-vertex graph.
    # The graph is self-complementary? No.
    
    # Let's try to generate a candidate and verify.
    # A common construction for large (5,5)-free graphs uses quadratic residues mod p where p is prime.
    # The largest prime < 42 is 41.
    # Paley graph on 41 vertices: 41 = 1 mod 4.
    # Let's check if Paley(41) is (5,5)-free.
    # If Paley(41) is (5,5)-free, then n=41 is valid.
    # The task says Exoo's 42-vertex graph proves R(5,5) >= 43.
    # So there is a 42-vertex graph.
    
    # Let's first try Paley(41).
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    n = q
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    
    if _verify_55_free(adj, n):
        return n, edges
    
    # If Paley(41) fails, we need the 42-vertex graph.
    # The 42-vertex graph is not a Paley graph.
    # I will attempt to construct a 42-vertex graph by adding a vertex to a 41-vertex graph or using a known list.
    # Since I don't have the exact edge list for Exoo's 42-graph memorized, and generating it from scratch is hard,
    # I will fall back to the best valid graph I can find quickly.
    
    # Let's try to search for a 42-vertex graph using a greedy/heuristic approach?
    # That might be too slow or unreliable.
    
    # Alternative: Use the Paley graph on 37 (known valid) as a safe fallback.
    # But we want to maximize n.
    
    # Let's try to see if we can extend Paley(37) or find a 41/42 graph.
    # Actually, let's try to verify Paley(41) more carefully.
    # If it's valid, 41 > 37, so it's an improvement.
    
    # What if Paley(41) is NOT (5,5)-free?
    # Then we stick to 37 or try to find another graph.
    
    # Let's try to find a 42-vertex graph by random search + local optimization?
    # No, that's risky.
    
    # Let's assume Paley(41) is valid. If not, we return 37.
    
    # Wait, I can try to hardcode the 42-vertex graph if I can find it online or derive it.
    # The 42-vertex graph is often referred to as the "Exoo graph".
    # It is a (42, 5, 5) graph.
    
    # Let's try a different approach: Use a random graph with p=0.5 and then remove edges to break K5 and Ind5?
    # No, that's complex.
    
    # Let's stick to Paley(41) if valid, else Paley(37).
    
    # Let's also try Paley(29), Paley(37), Paley(41).
    # Paley(37) is known to be (5,5)-free.
    # Paley(41) might be (5,5)-free.
    
    # Let's check Paley(41) validity.
    # If valid, return 41.
    
    # What about 42?
    # I will try to construct a 42-vertex graph by taking Paley(41) and adding a vertex connected to a specific subset.
    # Or, I can use a known list.
    
    # Since I cannot be 100% sure about the 42-vertex graph edge list without external data,
    # I will return the best valid graph found among Paley(37) and Paley(41).
    
    # Let's also try a quick local search to see if we can get 42?
    # No, too risky for the time limit.
    
    # Let's return Paley(41) if valid.
    
    # Actually, let's try to find a 42-vertex graph by using a SAT solver or ILP?
    # Not available.
    
    # I will return the best of Paley(37) and Paley(41).
    
    # Let's also check if there's a 42-vertex graph in the literature that I can hardcode.
    # The 42-vertex graph is a "strongly regular graph"? No.
    
    # Let's just return the best valid Paley graph.
    
    # Wait, I can try to use a heuristic to find a 42-vertex graph.
    # Start with a random graph on 42 vertices, p=0.5.
    # While there is a K5 or Ind5, remove/add edges to break it.
    # This is a local search.
    
    # Given the time constraint (330s), I should be careful.
    # Let's try a limited local search for 42 vertices.
    
    start_time = time.time()
    deadline = start_time + 300  # 300s safety margin
    
    # Try local search for 42 vertices
    n_target = 42
    best_n = 37
    best_edges = _paley_37()[1]
    
    # Check Paley(41)
    n_41, edges_41 = _paley_41()
    if _verify_55_free(_build_adjacency(n_41, edges_41), n_41):
        best_n = 41
        best_edges = edges_41
    
    # Try to improve to 42
    if time.time() < deadline:
        # Local search
        # Start with a random graph or extend Paley(41)
        # Let's start with a random graph
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            # Initialize random graph
            adj = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search: while there is a violation, fix it
            # We need to find a K5 or Ind5 and remove an edge from it (for K5) or add an edge (for Ind5).
            # This is complex to implement robustly.
            
            # Alternative: Use a greedy approach to build the graph.
            # Add vertices one by one, choosing connections to avoid K5 and Ind5.
            
            # Let's try a greedy construction for 42 vertices.
            adj = [0] * n_target
            valid = True
            for v in range(n_target):
                # Choose neighbors for v from {0, ..., v-1}
                # We want to avoid creating a K5 or Ind5.
                # For each potential neighbor u < v, check if adding edge (u,v) creates a K5.
                # And if NOT adding edge (u,v) creates an Ind5.
                
                # This is a constraint satisfaction problem.
                # Let's try a simple heuristic:
                # For each u < v, decide edge or no edge.
                # Prefer edge if it doesn't create K5, and no edge if it doesn't create Ind5.
                
                # This is getting complicated. Let's stick to the known valid graphs.
                pass
            
            # Since local search is risky, I will skip it and return the best known valid graph.
            
            break
    
    return best_n, best_edges

def _paley_41():
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

def _build_adjacency(n, edges):
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def construct():
    # Return the best valid graph
    # Try Paley(41) first
    n_41, edges_41 = _paley_41()
    adj_41 = _build_adjacency(n_41, edges_41)
    if _verify_55_free(adj_41, n_41):
        return n_41, edges_41
    
    # Fallback to Paley(37)
    n_37, edges_37 = _paley_37()
    return n_37, edges_37
# EVOLVE-BLOCK-END
