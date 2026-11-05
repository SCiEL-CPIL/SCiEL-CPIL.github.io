import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts for each n
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            # Precompute all 6-sets and their masks
            six_sets = list(combinations(range(n), 6))
            six_masks = [0] * len(six_sets)
            for i, s in enumerate(six_sets):
                m = 0
                for v in s:
                    m |= (1 << v)
                six_masks[i] = m
                
            # Map edge to list of 6-set indices containing it
            edge_to_sets = {}
            for i, s in enumerate(six_sets):
                for u in s:
                    for v in s:
                        if u < v:
                            if (u, v) not in edge_to_sets:
                                edge_to_sets[(u, v)] = []
                            edge_to_sets[(u, v)].append(i)
                            
            # Function to check if a 6-set is regular
            def is_regular(idx):
                s = six_sets[idx]
                m = six_masks[idx]
                degs = []
                for u in s:
                    # Count bits in adj[u] & m
                    degs.append((adj[u] & m).bit_count())
                return len(set(degs)) == 1
            
            # Initial check for regular 6-sets
            regular_six_idx = set()
            for i in range(len(six_sets)):
                if is_regular(i):
                    regular_six_idx.add(i)
                    
            # Local search
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                if not regular_six_idx:
                    # Check 7+ sets
                    violation_7 = False
                    # Sample some 7-sets to save time? Or check all?
                    # Checking all 7-sets is C(n,7). For n=22, C(22,7)=170544.
                    # This might be slow if done every step.
                    # Only do full 7+ check if 6-sets are clear.
                    # If we find a 7-violation, we need to break it.
                    
                    # To be safe, check all 7-sets
                    for s7 in combinations(range(n), 7):
                        m7 = 0
                        for v in s7:
                            m7 |= (1 << v)
                        degs = [(adj[u] & m7).bit_count() for u in s7]
                        if len(set(degs)) == 1:
                            violation_7 = True
                            # Flip a random edge in s7
                            u, v = random.sample(list(s7), 2)
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                            
                            # Update regular_six_idx for affected sets
                            affected = edge_to_sets.get((u, v), [])
                            for idx in affected:
                                was_reg = idx in regular_six_idx
                                is_reg = is_regular(idx)
                                if was_reg and not is_reg:
                                    regular_six_idx.remove(idx)
                                elif not was_reg and is_reg:
                                    regular_six_idx.add(idx)
                            break
                            
                    if not violation_7:
                        # Check 8+ sets?
                        # Probability is very low if 6 and 7 are clear.
                        # We can sample 8+ sets.
                        violation_8 = False
                        for _ in range(100):
                            s8 = random.sample(range(n), 8)
                            m8 = 0
                            for v in s8:
                                m8 |= (1 << v)
                            degs = [(adj[u] & m8).bit_count() for u in s8]
                            if len(set(degs)) == 1:
                                violation_8 = True
                                u, v = random.sample(s8, 2)
                                adj[u] ^= (1 << v)
                                adj[v] ^= (1 << u)
                                affected = edge_to_sets.get((u, v), [])
                                for idx in affected:
                                    was_reg = idx in regular_six_idx
                                    is_reg = is_regular(idx)
                                    if was_reg and not is_reg:
                                        regular_six_idx.remove(idx)
                                    elif not was_reg and is_reg:
                                        regular_six_idx.add(idx)
                                break
                                
                        if not violation_8:
                            # Success
                            edges = []
                            for u in range(n):
                                for v in range(u + 1, n):
                                    if (adj[u] >> v) & 1:
                                        edges.append([u, v])
                            best_n = n
                            best_edges = edges
                            found = True
                            break
                            
                else:
                    # Pick a random regular 6-set
                    idx = random.choice(list(regular_six_idx))
                    s = six_sets[idx]
                    u, v = random.sample(list(s), 2)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    # Update regular_six_idx
                    affected = edge_to_sets.get((u, v), [])
                    for a_idx in affected:
                        was_reg = a_idx in regular_six_idx
                        is_reg = is_regular(a_idx)
                        if was_reg and not is_reg:
                            regular_six_idx.remove(a_idx)
                        elif not was_reg and is_reg:
                            regular_six_idx.add(a_idx)
                            
            if found:
                break
                
    if best_n == 0:
        # Fallback: return a small valid graph, e.g., n=6? 
        # But n=6 must not have regular 6-subgraph.
        # A 6-cycle is regular (deg 2).
        # Path P6 is not regular.
        # So n=6 with edges of a path is valid.
        best_n = 6
        best_edges = [[i, i+1] for i in range(5)]
        
    return {'n': best_n, 'edges': best_edges}
