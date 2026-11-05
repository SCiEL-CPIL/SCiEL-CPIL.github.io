"""
K3,3-free graph construction on 20 vertices.
Strategy: Start from a dense random graph, then iteratively remove edges
that create K3,3 subgraphs. Use local search (edge swaps) to improve.
"""

import time
import random
import itertools
from collections import defaultdict

def _count_common_neighbors(adj, u, v):
    """Count common neighbors of u and v in adjacency matrix."""
    # adj is a list of sets or a 2D array
    if isinstance(adj[0], set):
        return len(adj[u] & adj[v])
    else:
        # numpy or list of lists
        common = 0
        for w in range(len(adj)):
            if adj[u][w] and adj[v][w]:
                common += 1
        return common

def _find_k33_violations(adj, n=20):
    """
    Find all triples (a,b,c) that have 3+ common neighbors.
    Returns list of (a,b,c) tuples.
    """
    violations = []
    for a in range(n):
        for b in range(a+1, n):
            common_ab = adj[a] & adj[b] if isinstance(adj[0], set) else set(i for i in range(n) if adj[a][i] and adj[b][i])
            for c in range(b+1, n):
                common_abc = common_ab & adj[c] if isinstance(adj[0], set) else common_ab & set(i for i in range(n) if adj[c][i])
                if len(common_abc) >= 3:
                    violations.append((a, b, c))
    return violations

def _is_valid(adj, n=20):
    """Check if graph has no K3,3 (no three vertices with 3+ common neighbors)."""
    for a in range(n):
        for b in range(a+1, n):
            common_ab = adj[a] & adj[b] if isinstance(adj[0], set) else set(i for i in range(n) if adj[a][i] and adj[b][i])
            for c in range(b+1, n):
                if len(common_ab & adj[c]) >= 3 if isinstance(adj[0], set) else len(common_ab & set(i for i in range(n) if adj[c][i])) >= 3:
                    return False
    return True

def _edge_count(adj, n=20):
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                count += 1
    return count

def _edge_count_set(adj, n=20):
    return sum(len(s) for s in adj) // 2

def construct():
    n = 20
    deadline = time.time() + 120  # 120 second deadline
    best_edges = []
    best_count = 0
    
    def to_edge_list(adj_matrix):
        """Convert adjacency matrix (list of sets) to edge list."""
        edges = []
        for i in range(n):
            for j in adj_matrix[i]:
                if i < j:
                    edges.append([i, j])
        return edges
    
    def validate_and_get_edges(adj):
        """Validate graph and return edge list if valid."""
        if not _is_valid(adj, n):
            return None, 0
        edges = to_edge_list(adj)
        return edges, len(edges)
    
    # Try multiple random initializations with repair
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        
        # Start with a dense random graph
        p = 0.6  # probability of edge
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Repair: iteratively remove edges that create K3,3
        max_repair_iterations = 1000
        for iter_i in range(max_repair_iterations):
            if time.time() > deadline - 30:
                break
            
            violations = _find_k33_violations(adj, n)
            if not violations:
                break
            
            # For each violation, try to remove an edge that breaks it
            fixed = False
            for (a, b, c) in violations:
                if time.time() > deadline - 30:
                    break
                
                common_abc = adj[a] & adj[b] & adj[c]
                if len(common_abc) < 3:
                    continue
                
                # Try removing an edge from one of the common neighbors
                # to break the K3,3
                for x in list(common_abc):
                    # Try removing edge (a, x)
                    adj[a].discard(x)
                    adj[x].discard(a)
                    if _is_valid(adj, n):
                        fixed = True
                        break
                    else:
                        adj[a].add(x)
                        adj[x].add(a)
                
                if fixed:
                    break
            
            if not fixed:
                # Try removing random edges
                all_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                random.shuffle(all_edges)
                removed = False
                for (i, j) in all_edges:
                    adj[i].discard(j)
                    adj[j].discard(i)
                    if _is_valid(adj, n):
                        removed = True
                        break
                    else:
                        adj[i].add(j)
                        adj[j].add(i)
                
                if not removed:
                    break
        
        # Local search: try to add edges
        current_count = _edge_count_set(adj, n)
        improved = True
        max_local_iters = 500
        local_iter = 0
        while improved and local_iter < max_local_iters:
            if time.time() > deadline - 30:
                break
            
            improved = False
            local_iter += 1
            
            # Try adding each missing edge
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        # Try adding edge (i, j)
                        adj[i].add(j)
                        adj[j].add(i)
                        if _is_valid(adj, n):
                            current_count += 1
                            improved = True
                        else:
                            adj[i].discard(j)
                            adj[j].discard(i)
        
        edges, count = validate_and_get_edges(adj)
        if edges is not None and count > best_count:
            best_count = count
            best_edges = edges
    
    # If we found a good graph, try to improve it further with more aggressive local search
    if best_edges:
        if time.time() < deadline - 30:
            # Reconstruct adjacency from best_edges
            adj = [set() for _ in range(n)]
            for e in best_edges:
                u, v = e[0], e[1]
                adj[u].add(v)
                adj[v].add(u)
            
            # More aggressive local search with edge swaps
            current_count = len(best_edges)
            max_swap_iters = 2000
            for swap_iter in range(max_swap_iters):
                if time.time() > deadline - 30:
                    break
                
                improved = False
                
                # Try adding edges
                for i in range(n):
                    for j in range(i+1, n):
                        if j not in adj[i]:
                            adj[i].add(j)
                            adj[j].add(i)
                            if _is_valid(adj, n):
                                current_count += 1
                                improved = True
                            else:
                                adj[i].discard(j)
                                adj[j].discard(i)
                
                if improved:
                    continue
                
                # Try edge swaps: remove one edge, add another
                all_existing = [(i, j) for i in range(n) for j in adj[i] if i < j]
                all_missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
                
                random.shuffle(all_existing)
                random.shuffle(all_missing)
                
                swap_found = False
                for (rem_i, rem_j) in all_existing[:50]:
                    if time.time() > deadline - 30:
                        break
                    for (add_i, add_j) in all_missing[:50]:
                        if (rem_i, rem_j) == (add_i, add_j):
                            continue
                        # Remove (rem_i, rem_j), add (add_i, add_j)
                        adj[rem_i].discard(rem_j)
                        adj[rem_j].discard(rem_i)
                        adj[add_i].add(add_j)
                        adj[add_j].add(add_i)
                        if _is_valid(adj, n):
                            # Count edges
                            new_count = _edge_count_set(adj, n)
                            if new_count > current_count:
                                current_count = new_count
                                swap_found = True
                                improved = True
                                break
                            else:
                                # Revert
                                adj[rem_i].add(rem_j)
                                adj[rem_j].add(rem_i)
                                adj[add_i].discard(add_j)
                                adj[add_j].discard(add_i)
                        else:
                            # Revert
                            adj[rem_i].add(rem_j)
                            adj[rem_j].add(rem_i)
                            adj[add_i].discard(add_j)
                            adj[add_j].discard(add_i)
                    
                    if swap_found:
                        break
                
                if not improved:
                    break
            
            edges, count = validate_and_get_edges(adj)
            if edges is not None and count > best_count:
                best_count = count
                best_edges = edges
    
    # Final fallback: if best is still low, use a known construction
    if best_count < 100:
        # Use a simpler construction: complement of a perfect matching + some edges
        # Actually, let's just return what we have
        pass
    
    return best_edges
