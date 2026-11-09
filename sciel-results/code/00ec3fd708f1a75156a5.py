import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    n = 15
    
    def make_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def is_valid(adj):
        """Check: no path s-u-v-t with deg(s)==deg(t), s!=t"""
        degs = [len(adj[i]) for i in range(n)]
        # For each pair of vertices with same degree, check they have no common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                if degs[i] == degs[j]:
                    # Check if i and j share a common neighbor
                    common = adj[i] & adj[j]
                    if common:
                        return False
        return True
    
    def is_valid_fast(adj, degs):
        """Faster check using precomputed degrees"""
        for i in range(n):
            for j in range(i + 1, n):
                if degs[i] == degs[j]:
                    common = adj[i] & adj[j]
                    if common:
                        return False
        return True
    
    def count_violations(adj, degs):
        """Count number of violating pairs (for soft objective)"""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if degs[i] == degs[j]:
                    if adj[i] & adj[j]:
                        count += 1
        return count
    
    # Start with K_{7,8}
    A = list(range(7))   # 0..6
    B = list(range(7, 15))  # 7..14
    
    base_edges = []
    for a in A:
        for b in B:
            base_edges.append([a, b])
    
    adj = make_adj(base_edges)
    
    # Verify base is valid
    if not is_valid(adj):
        # Fallback to star
        return [[0, v] for v in range(1, n)]
    
    best_edges = list(base_edges)
    best_count = len(base_edges)
    
    # Greedy: try adding each candidate edge if it maintains validity
    # Build candidate edges (within-partition edges)
    candidate_edges = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidate_edges.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidate_edges.append((B[i], B[j]))
    
    random.shuffle(candidate_edges)
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        if v in adj[u]:
            continue
        # Try adding
        adj[u].add(v)
        adj[v].add(u)
        if is_valid(adj):
            best_edges.append([u, v])
            best_count += 1
        else:
            adj[u].remove(v)
            adj[v].remove(u)
    
    # Multi-restart local search with remove-then-add
    best_global = list(best_edges)
    best_global_count = best_count
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Start from current best or a perturbed version
        current_adj = make_adj(best_global)
        current_edges = list(best_global)
        current_count = len(current_edges)
        
        # Perturbation: remove some random edges
        if restart > 0:
            num_removals = random.randint(2, 5)
            removable = list(current_edges)
            random.shuffle(removable)
            for _ in range(min(num_removals, len(removable))):
                u, v = removable.pop()
                if v in current_adj[u]:
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
                    current_count -= 1
            current_edges = list(current_adj)
            # Rebuild edge list properly
            current_edges = []
            for u in range(n):
                for v in current_adj[u]:
                    if u < v:
                        current_edges.append([u, v])
            current_count = len(current_edges)
        
        # Local search: repeatedly try to add edges
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Get all missing edges
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if j not in current_adj[i]:
                        missing.append((i, j))
            
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                if v in current_adj[u]:
                    continue
                
                # Try adding
                current_adj[u].add(v)
                current_adj[v].add(u)
                
                if is_valid(current_adj):
                    current_count += 1
                    improved = True
                else:
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
        
        # Update best
        if current_count > best_global_count:
            best_global = []
            for u in range(n):
                for v in current_adj[u]:
                    if u < v:
                        best_global.append([u, v])
            best_global_count = current_count
    
    # Final: try one more pass on best
    final_adj = make_adj(best_global)
    for i in range(n):
        if time.time() > deadline - 2:
            break
        for j in range(i+1, n):
            if time.time() > deadline - 2:
                break
            if j not in final_adj[i]:
                final_adj[i].add(j)
                final_adj[j].add(i)
                if is_valid(final_adj):
                    best_global.append([i, j])
                else:
                    final_adj[i].remove(j)
                    final_adj[j].remove(i)
    
    # Convert to proper list of lists with int types
    result = []
    for e in best_global:
        u, v = e
        result.append([int(u), int(v)])
    
    return result
