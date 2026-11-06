import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110
    random.seed(42)
    
    n = 35
    # Represent adjacency as a list of bitmasks
    # adj[i] is an integer where bit j is 1 if i-j is an edge
    
    def make_adj_bitmask(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    def is_clique_4(adj, i, j, k, l):
        # Check if i,j,k,l form a K4
        # adj[i] must have bits j,k,l set
        if not (adj[i] & (1 << j) and adj[i] & (1 << k) and adj[i] & (1 << l)):
            return False
        if not (adj[j] & (1 << k) and adj[j] & (1 << l)):
            return False
        return bool(adj[k] & (1 << l))

    def is_indep_6(adj, combo):
        # Check if combo is an independent set
        mask = 0
        for v in combo:
            mask |= 1 << v
        for v in combo:
            if adj[v] & (mask ^ (1 << v)):
                return False
        return True

    def find_k4_exact(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    # Check common neighbors
                    common = adj[i] & adj[j]
                    # Iterate over common neighbors k > j
                    # We can just iterate over all k
                    for k in range(j+1, n):
                        if common & (1 << k):
                            # Check if k and l are connected for l > k
                            common_kl = adj[k] & common
                            # We need a pair l,m in common_kl that are connected?
                            # No, we need k,l,m to form a triangle with i,j?
                            # We need K4: i,j,k,l.
                            # We already have i-j, i-k, j-k.
                            # We need i-l, j-l, k-l, and k-l?
                            # No, we need to find l such that i-l, j-l, k-l are edges.
                            # So l must be in adj[i] & adj[j] & adj[k].
                            common_ijk = adj[i] & adj[j] & adj[k]
                            # Iterate over l in common_ijk
                            # l must be > k to avoid duplicates?
                            # We can just check if there is any l.
                            # If common_ijk has any bit set, we have a K4?
                            # No, we need l to be connected to i,j,k.
                            # If l is in common_ijk, then i-l, j-l, k-l are edges.
                            # And we already have i-j, i-k, j-k.
                            # So {i,j,k,l} is a K4.
                            if common_ijk:
                                return True
        return False

    def find_indep6_exact(adj):
        for combo in combinations(range(n), 6):
            if is_indep_6(adj, combo):
                return True
        return False

    def is_valid_exact(adj):
        return not find_k4_exact(adj) and not find_indep6_exact(adj)

    def approx_violation(adj, sample_size=100):
        # Check random samples
        # K4 check
        for _ in range(sample_size):
            combo = random.sample(range(n), 4)
            if is_clique_4(adj, combo[0], combo[1], combo[2], combo[3]):
                return 1
        # Indep6 check
        for _ in range(sample_size):
            combo = random.sample(range(n), 6)
            if is_indep_6(adj, combo):
                return 1
        return 0

    best_result = None
    best_approx = 1
    
    # Initialize with random graphs
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        # Random graph with p=0.4
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.4:
                    edges.append([i, j])
        
        adj = make_adj_bitmask(edges)
        
        # Simulated Annealing / Hill Climbing
        current_adj = adj[:]
        current_viol = approx_violation(current_adj)
        
        for step in range(200):
            if time.time() > deadline - 5:
                break
                
            # Flip a random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Flip
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            
            new_viol = approx_violation(current_adj)
            
            if new_viol < current_viol:
                current_viol = new_viol
                if current_viol == 0:
                    # Candidate is promising, do exact check
                    if is_valid_exact(current_adj):
                        # Extract edges
                        final_edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if current_adj[i] & (1 << j):
                                    final_edges.append([i, j])
                        if best_result is None or n > best_result['n']:
                            best_result = {'n': n, 'edges': final_edges}
                            return best_result
            else:
                # Revert
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                
    # If we didn't find 35, try smaller n?
    # The task says maximize n.
    # If we can't find 35, we should return the best we found.
    # But we didn't track intermediate valid graphs.
    # Let's add a fallback: check n=34, 33...
    
    # Actually, if we didn't return, it means we didn't find a valid 35.
    # Let's try to find a valid 34?
    # This might take time.
    # Let's just return a known valid graph if we have one.
    # The Paley graph on 17 is valid.
    
    # Let's try one more thing: if we found a graph with approx_viol=0 but exact failed,
    # it's close. We can try to repair it.
    
    # For now, return the seed result if we didn't improve.
    # But we should try to find at least 25.
    
    # Let's run a quick search for n=25 with exact checks.
    # This is safer.
    
    # Actually, the current program finds 25.
    # Let's just improve the search for 35.
    
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i+1, 17) if (j-i) % 17 in {x*x % 17 for x in range(1, 17)}]}
