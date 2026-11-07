import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    n = 15
    best_edges = []
    best_score = -1

    def get_edges_list(edge_set):
        return [[u, v] for u, v in edge_set]

    def calculate_degrees(edge_set, n):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg

    def is_valid(edge_set, n):
        deg = calculate_degrees(edge_set, n)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check all 3-paths s-u-v-t
        for u, v in edge_set:
            for s in adj[u]:
                if s == v: continue
                for t in adj[v]:
                    if t == u: continue
                    if s == t: continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def find_violation(edge_set, n):
        deg = calculate_degrees(edge_set, n)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in edge_set:
            for s in adj[u]:
                if s == v: continue
                for t in adj[v]:
                    if t == u: continue
                    if s == t: continue
                    if deg[s] == deg[t]:
                        return (s, u, v, t)
        return None

    def try_build_from_sequence(deg_seq):
        # Greedy construction
        edges = set()
        remaining = list(deg_seq)
        # Map original index to current remaining degree
        # We need to track which vertex is which
        # Let's say vertex i has target degree deg_seq[i]
        
        # Sort vertices by remaining degree, descending
        # We will connect the vertex with highest remaining degree to others with highest remaining degree
        
        # This is a simplified Havel-Hakimi or similar
        # But we need to return a set of edges on 0..14
        
        # Let's use a simple greedy:
        # Sort vertices by deg_seq descending.
        # For i in range(n):
        #   Connect i to the next deg_seq[i] vertices that have the highest remaining capacity?
        
        # Actually, let's just use a random greedy that respects degrees approximately.
        # Or better, use a known algorithm for graphical sequence realization if available, 
        # but we can write a simple one.
        
        # Simple Greedy:
        # 1. Create a list of (degree, vertex)
        # 2. Sort by degree descending
        # 3. For the first vertex v, connect it to the next 'degree' vertices in the list
        # 4. Decrease their degrees
        # 5. Repeat
        # This might fail if the sequence is not graphical or if greedy choices block later vertices.
        
        # Let's try multiple random orderings for the greedy steps.
        
        for _ in range(10): # 10 random restarts of the greedy builder
            if time.time() > deadline:
                return None
            
            rem_deg = list(deg_seq)
            edges = set()
            success = True
            
            # We need to be careful with the order.
            # Let's pick vertices in random order but prioritize those with high remaining degree?
            # Or just standard greedy: highest degree first.
            
            # To add randomness, we can shuffle ties or pick from top K.
            
            for _step in range(n):
                if time.time() > deadline:
                    return None
                # Find vertex with max remaining degree
                # If all zero, break
                max_deg = max(rem_deg)
                if max_deg == 0:
                    break
                
                # Pick a random vertex among those with max_deg
                candidates = [i for i in range(n) if rem_deg[i] == max_deg]
                u = random.choice(candidates)
                
                # Connect u to 'max_deg' other vertices with highest remaining degrees
                # Exclude u
                others = [(rem_deg[j], j) for j in range(n) if j != u and rem_deg[j] > 0]
                others.sort(reverse=True)
                
                if len(others) < max_deg:
                    success = False
                    break
                
                # Pick the top max_deg vertices
                # Add randomness: pick from top max_deg + 2?
                k = max_deg
                if len(others) > k:
                    # Shuffle the top k+5 to add variety
                    top_k_plus = others[:k+5]
                    random.shuffle(top_k_plus)
                    targets = [v for d, v in top_k_plus[:k]]
                else:
                    targets = [v for d, v in others]
                
                for v in targets:
                    if (min(u, v), max(u, v)) in edges:
                        continue
                    edges.add((min(u, v), max(u, v)))
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
                
                # Check if rem_deg[u] is 0
                if rem_deg[u] != 0:
                    success = False
                    break
            
            if success and all(d == 0 for d in rem_deg):
                return edges
        
        return None

    # Define some target degree sequences
    # 1. All distinct except one pair
    # Sum must be even.
    # 0..14 sum is 105 (odd).
    # Change 0 to 1? 1..14 sum is 104 (even). But then we have 1,1,2..14.
    # Degrees: 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14.
    seq1 = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # 2. Another sequence
    # 0, 2, 3, ..., 14? Sum = 105 - 1 = 104 (even).
    seq2 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14] # Sum = 104? 2+3+..+14 = 103. 103+14=117 (odd).
    # Let's calculate:
    # 1+2+...+14 = 105.
    # To make it even, we can change one number by 1.
    # e.g., 1->2: 2,2,3..14. Sum = 106.
    seq3 = [2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14] # Wait, 14 is used twice?
    # Indices: 0:2, 1:2, 2:3, ..., 13:14, 14:14.
    # Sum: 2+2+3+...+13+14+14.
    # 3+...+13 = (3+13)*11/2 = 88.
    # 2+2+88+14+14 = 120. Even.
    
    sequences = [seq1, seq3]
    
    # Also try random sequences with few duplicates
    for _ in range(50):
        if time.time() > deadline:
            break
        seq = list(range(1, 15)) # 1..14
        # Add one more degree
        # Pick a random degree to duplicate
        dup_val = random.choice(seq)
        seq.append(dup_val)
        # Shuffle
        random.shuffle(seq)
        # Check sum parity
        if sum(seq) % 2 != 0:
            # Change one degree by 1
            idx = random.randint(0, 14)
            if seq[idx] < 14:
                seq[idx] += 1
            elif seq[idx] > 0:
                seq[idx] -= 1
        sequences.append(seq)

    for seq in sequences:
        if time.time() > deadline:
            break
        
        edges = try_build_from_sequence(seq)
        if edges is None:
            continue
        
        # Local Search to fix validity
        # If not valid, try to remove edges from violating paths
        steps = 0
        while steps < 100:
            if time.time() > deadline:
                break
            if is_valid(edges, n):
                break
            
            viol = find_violation(edges, n)
            if viol is None:
                break
            
            s, u, v, t = viol
            # Remove a random edge from the path or incident to s,t
            candidates = [
                (min(s, u), max(s, u)),
                (min(u, v), max(u, v)),
                (min(v, t), max(v, t))
            ]
            # Also consider removing other edges to change degrees
            # But removing path edges is most direct
            valid_candidates = [e for e in candidates if e in edges]
            if not valid_candidates:
                # Fallback: remove any edge
                valid_candidates = list(edges)
            
            e = random.choice(valid_candidates)
            edges.remove(e)
            steps += 1
        
        if is_valid(edges, n):
            score = len(edges)
            if score > best_score:
                best_score = score
                best_edges = get_edges_list(edges)

    # Final local search improvement on best_edges
    if best_edges:
        current_edges = set(tuple(e) for e in best_edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            all_edges = set()
            for i in range(n):
                for j in range(i+1, n):
                    all_edges.add((i, j))
            missing = list(all_edges - current_edges)
            random.shuffle(missing)
            
            for e in missing:
                if time.time() > deadline:
                    break
                current_edges.add(e)
                if is_valid(current_edges, n):
                    improved = True
                else:
                    current_edges.remove(e)
        
        if len(current_edges) > len(best_edges):
            best_edges = get_edges_list(current_edges)

    return best_edges
