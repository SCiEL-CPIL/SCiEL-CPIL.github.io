"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
import sys
import numpy as np
from scipy.special import comb as scipy_comb

def construct():
    """
    Construct a graph with no K5 and no independent set of size 5.
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Start with Paley graph on 37 vertices as seed
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_seed = q
    adj_seed = np.zeros((n_seed, n_seed), dtype=bool)
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            if (j - i) % q in residues:
                adj_seed[i, j] = True
                adj_seed[j, i] = True
    
    # Verify seed is valid (5,5)-free
    if not _is_55_free(adj_seed, n_seed):
        # Fallback: just return seed
        return _adj_to_edges(adj_seed, n_seed)
    
    best_n = n_seed
    best_adj = adj_seed.copy()
    
    # Try to extend by adding vertices one at a time
    # Strategy: try adding a vertex connected to a random subset of existing vertices
    # that maintains the (5,5)-free property
    
    random.seed(42)
    
    current_n = n_seed
    current_adj = adj_seed.copy()
    
    # Try to extend up to 42 vertices (Exoo's bound)
    target_n = 42
    max_extensions = target_n - current_n
    
    for extension in range(max_extensions):
        if time.time() > deadline - 30:  # Stop if approaching deadline
            break
        
        new_n = current_n + 1
        success = False
        
        # Try multiple random connection patterns
        for attempt in range(500):
            if time.time() > deadline - 30:
                break
            
            # Randomly decide connections from new vertex to existing ones
            # Use a heuristic: connect to vertices that don't create K5 or independent 5
            new_connections = np.zeros(current_n, dtype=bool)
            
            # Try to balance: connect to roughly half the vertices
            # Use a greedy approach with some randomness
            for v in range(current_n):
                # Check if connecting v would create a K5
                # A K5 would require 4 other vertices all connected to new vertex and forming K4 among themselves
                # This is complex, so use a simpler heuristic:
                # Connect if the number of common neighbors with any 4-set doesn't create issues
                
                # Simple heuristic: connect with probability based on degree
                # Vertices with lower degree get higher connection probability
                deg_v = np.sum(current_adj[v])
                prob = 0.5 * (1 - deg_v / current_n) + 0.3
                if random.random() < prob:
                    new_connections[v] = True
            
            # Build candidate adjacency matrix
            candidate_adj = np.zeros((new_n, new_n), dtype=bool)
            candidate_adj[:current_n, :current_n] = current_adj
            candidate_adj[current_n, :current_n] = new_connections
            candidate_adj[:current_n, current_n] = new_connections
            
            # Check if (5,5)-free
            if _is_55_free(candidate_adj, new_n):
                current_n = new_n
                current_adj = candidate_adj
                success = True
                break
        
        if not success:
            # Try a different strategy: start over with different random seed
            # or try to find a better connection pattern
            if time.time() > deadline - 20:
                break
            
            # Try systematic approach: try all subsets of size ~current_n/2
            # This is expensive, so limit attempts
            found = False
            for attempt2 in range(100):
                if time.time() > deadline - 20:
                    break
                
                # Generate a random subset of size current_n // 2
                k = random.randint(current_n // 3, 2 * current_n // 3)
                connected_set = set(random.sample(range(current_n), k))
                
                new_connections2 = np.array([v in connected_set for v in range(current_n)])
                
                candidate_adj2 = np.zeros((new_n, new_n), dtype=bool)
                candidate_adj2[:current_n, :current_n] = current_adj
                candidate_adj2[current_n, :current_n] = new_connections2
                candidate_adj2[:current_n, current_n] = new_connections2
                
                if _is_55_free(candidate_adj2, new_n):
                    current_n = new_n
                    current_adj = candidate_adj2
                    found = True
                    break
            
            if not found:
                break  # Can't extend further
    
    # Return the best graph found
    final_n = current_n
    final_adj = current_adj
    
    # Ensure we have at least the seed
    if final_n < best_n:
        final_n = best_n
        final_adj = best_adj
    
    edges = _adj_to_edges(final_adj, final_n)
    return final_n, edges


def _adj_to_edges(adj, n):
    """Convert adjacency matrix to list of edges."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append((i, j))
    return edges


def _is_55_free(adj, n):
    """
    Check if graph has no clique of size 5 and no independent set of size 5.
    Uses bitset-based clique/independent set detection.
    """
    if n < 5:
        return True
    
    # Build bitsets for neighbors and non-neighbors
    # For each vertex, precompute the set of neighbors as a bitset
    neighbor_bits = []
    non_neighbor_bits = []
    for i in range(n):
        nb = 0
        nnb = 0
        for j in range(n):
            if i != j:
                if adj[i, j]:
                    nb |= (1 << j)
                else:
                    nnb |= (1 << j)
        neighbor_bits.append(nb)
        non_neighbor_bits.append(nnb)
    
    # Check for K5: for each 5-subset, check if all pairs are connected
    # More efficient: for each vertex, check if its neighborhood contains a K4
    # For each vertex i, check if there's a K4 in N(i)
    for i in range(n):
        # Get neighbors of i
        neighbors = [j for j in range(n) if j != i and adj[i, j]]
        if len(neighbors) < 4:
            continue
        
        # Check if neighbors contain a K4
        # For each 4-subset of neighbors, check if all pairs are connected
        for combo in combinations(neighbors, 4):
            is_k4 = True
            for a, b in combinations(combo, 2):
                if not adj[a, b]:
                    is_k4 = False
                    break
            if is_k4:
                return False  # Found K5
    
    # Check for independent set of size 5
    # For each vertex, check if its non-neighborhood contains an independent 4-set
    # which is equivalent to a K4 in the complement
    for i in range(n):
        # Get non-neighbors of i (excluding i itself)
        non_neighbors = [j for j in range(n) if j != i and not adj[i, j]]
        if len(non_neighbors) < 4:
            continue
        
        # Check if non_neighbors contain a K4 in the complement (i.e., independent set of size 4 in original)
        # This means: for all pairs in the 4-set, they are NOT connected in original
        for combo in combinations(non_neighbors, 4):
            is_independent_4 = True
            for a, b in combinations(combo, 2):
                if adj[a, b]:
                    is_independent_4 = False
                    break
            if is_independent_4:
                return False  # Found independent set of size 5
    
    return True


def _is_55_free_fast(adj, n):
    """
    Faster check using bit operations.
    """
    if n < 5:
        return True
    
    # For each vertex, compute neighbor bitset
    neighbor_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if i != j and adj[i, j]:
                bits |= (1 << j)
        neighbor_bits[i] = bits
    
    # Check for K5
    for i in range(n):
        # Count how many K4s exist in N(i)
        # A K4 in N(i) means 4 vertices all mutually adjacent
        neighbors = [j for j in range(n) if j != i and adj[i, j]]
        if len(neighbors) < 4:
            continue
        
        # For each pair of neighbors, check if they have common neighbors forming a K4
        # More efficient: for each 4-subset, check all 6 edges
        for a, b, c, d in combinations(neighbors, 4):
            if (adj[a, b] and adj[a, c] and adj[a, d] and 
                adj[b, c] and adj[b, d] and adj[c, d]):
                return False
    
    # Check for independent 5-set
    for i in range(n):
        non_neighbors = [j for j in range(n) if j != i and not adj[i, j]]
        if len(non_neighbors) < 4:
            continue
        
        for a, b, c, d in combinations(non_neighbors, 4):
            if (not adj[a, b] and not adj[a, c] and not adj[a, d] and 
                not adj[b, c] and not adj[b, d] and not adj[c, d]):
                return False
    
    return True
# EVOLVE-BLOCK-END
