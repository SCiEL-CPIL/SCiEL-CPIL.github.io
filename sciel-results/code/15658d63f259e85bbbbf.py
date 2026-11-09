import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    def is_regular_mask(mask, adj, n):
        """Check if induced subgraph on mask is regular (all vertices same degree within mask)."""
        verts = [v for v in range(n) if mask & (1 << v)]
        k = len(verts)
        if k < 6:
            return False
        degs = set()
        for v in verts:
            degs.add((adj[v] & mask).bit_count())
            if len(degs) > 1:
                return False
        return True
    
    def count_violations(mask, adj, n):
        """Count regular induced subgraphs in mask (for greedy scoring)."""
        # Only count subsets of size >= 6 within mask
        verts = [v for v in range(n) if mask & (1 << v)]
        k = len(verts)
        if k < 6:
            return 0
        count = 0
        # Check all subsets of size 6 from verts
        for comb in itertools.combinations(range(k), 6):
            submask = 0
            for idx in comb:
                submask |= 1 << verts[idx]
            if is_regular_mask(submask, adj, n):
                count += 1
        # For efficiency in greedy, we mainly care about size 6
        # Size 7+ violations are rarer and harder to create incrementally
        return count
    
    def full_verify(adj, n):
        """Full verification: no regular induced subgraph on >= 6 vertices."""
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= 1 << v
                if is_regular_mask(mask, adj, n):
                    return False
        return True
    
    def greedy_construct(target_n, time_limit):
        """Greedy sequential vertex addition."""
        adj = [0] * target_n
        
        for k in range(target_n):
            if time.time() > time_limit:
                break
            if k == 0:
                continue
            if k == 1:
                # Two vertices: no 6-subset possible, just pick random
                if random.random() < 0.5:
                    adj[0] |= 1 << 1
                    adj[1] |= 1 << 0
                continue
            
            # Try all 2^k possible connection patterns for vertex k
            # But that's too many for large k. Use sampling + greedy.
            
            best_pattern = None
            best_score = float('inf')
            
            # For small k, try all patterns. For large k, sample.
            num_patterns = 1 << k
            if num_patterns <= 2048:
                patterns = range(num_patterns)
            else:
                # Sample patterns
                patterns = [random.getrandbits(k) for _ in range(2048)]
                # Also include some structured patterns
                patterns.append(0)  # no edges
                patterns.append((1 << k) - 1)  # all edges
                patterns.append(random.getrandbits(k))
            
            for pattern in patterns:
                # Apply pattern
                for i in range(k):
                    if pattern & (1 << i):
                        adj[k] |= 1 << i
                        adj[i] |= 1 << k
                    else:
                        adj[k] &= ~(1 << i)
                        adj[i] &= ~(1 << k)
                
                # Score: count new regular 6-subsets that include vertex k
                score = 0
                # Subsets of size 6 that include k: choose 5 from {0..k-1}
                if k >= 5:
                    for comb in itertools.combinations(range(k), 5):
                        mask = 1 << k
                        for v in comb:
                            mask |= 1 << v
                        if is_regular_mask(mask, adj, target_n):
                            score += 1
                
                # Also check size 7 subsets including k (less critical but important)
                if k >= 6 and score == 0:
                    for comb in itertools.combinations(range(k), 6):
                        mask = 1 << k
                        for v in comb:
                            mask |= 1 << v
                        if is_regular_mask(mask, adj, target_n):
                            score += 1
                
                # Undo pattern
                for i in range(k):
                    if pattern & (1 << i):
                        adj[k] &= ~(1 << i)
                        adj[i] &= ~(1 << k)
                    else:
                        adj[k] |= 1 << i
                        adj[i] |= 1 << k
                
                if score < best_score:
                    best_score = score
                    best_pattern = pattern
                    if score == 0:
                        break
            
            # Apply best pattern
            if best_pattern is not None:
                for i in range(k):
                    if best_pattern & (1 << i):
                        adj[k] |= 1 << i
                        adj[i] |= 1 << k
                    else:
                        adj[k] &= ~(1 << i)
                        adj[i] &= ~(1 << k)
        
        return adj
    
    def local_search(adj, n, time_limit):
        """Local search to eliminate remaining regular subgraphs."""
        # Find all violating subsets
        def find_violations():
            violations = []
            for k in range(6, n + 1):
                for comb in itertools.combinations(range(n), k):
                    mask = 0
                    for v in comb:
                        mask |= 1 << v
                    if is_regular_mask(mask, adj, n):
                        violations.append(mask)
            return violations
        
        violations = find_violations()
        
        while violations and time.time() < time_limit:
            # Pick a random violation
            mask = random.choice(violations)
            verts = [v for v in range(n) if mask & (1 << v)]
            
            # Pick a random edge to flip within this subset
            u, v = random.sample(verts, 2)
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            
            # Re-find violations (expensive but necessary for correctness)
            violations = find_violations()
        
        return adj, len(violations) == 0
    
    # Try greedy construction for n = 22, 21, 20
    for target_n in [22, 21, 20]:
        if time.time() > deadline - 20:
            break
        
        # Multiple attempts
        best_adj = None
        best_n = 0
        
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            random.seed(attempt * 12345 + target_n)
            adj = greedy_construct(target_n, deadline - 20)
            
            # Verify
            if full_verify(adj, target_n):
                best_adj = adj
                best_n = target_n
                break
            
            # Try local search to fix
            adj2, fixed = local_search(adj, target_n, deadline - 20)
            if fixed and full_verify(adj2, target_n):
                best_adj = adj2
                best_n = target_n
                break
        
        if best_n > 0:
            adj = best_adj
            n = best_n
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
