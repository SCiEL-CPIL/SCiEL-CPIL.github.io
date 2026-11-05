<<<<<<< SEARCH
    def greedy_initial_decomposition(edges, seed):
        """
        Create an initial path decomposition using greedy approach.
        Returns list of paths.
        """
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        
        while remaining:
            # Pick a random edge to start
            edge = rng.choice(list(remaining))
            remaining.remove(edge)
            u, v = edge
            path = [u, v]
            
            # Extend the path greedily
            changed = True
            while changed:
                changed = False
                # Try extending from both ends
                for end in [0, 1]:  # 0 = front, 1 = back
                    if end == 0:
                        current = path[0]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if candidates:
                            nb = rng.choice(candidates)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.insert(0, nb)
                            changed = True
                    else:
                        current = path[-1]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if candidates:
                            nb = rng.choice(candidates)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.append(nb)
                            changed = True
                
                if not changed:
                    break
            
            paths.append(path)
        
        return paths
=======
    def greedy_initial_decomposition(edges, seed):
        """
        Create an initial path decomposition using greedy approach with degree-based heuristic.
        Prefer starting from low-degree vertices and extending toward high-degree vertices.
        Returns list of paths.
        """
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        paths = []
        
        while remaining:
            # Pick an edge to start - prefer edges where both endpoints have low degree
            # This tends to create longer paths
            candidates = list(remaining)
            if len(candidates) > 1:
                # Score edges by sum of degrees (lower is better for starting)
                candidates.sort(key=lambda e: degrees[e[0]] + degrees[e[1]])
                # Pick from the lowest 20% or random from all if small
                if len(candidates) > 5:
                    edge = rng.choice(candidates[:max(1, len(candidates) // 5)])
                else:
                    edge = rng.choice(candidates)
            else:
                edge = candidates[0]
            
            remaining.remove(edge)
            u, v = edge
            path = [u, v]
            
            # Extend the path greedily - prefer extending from the end with more options
            changed = True
            while changed:
                changed = False
                # Try extending from both ends, prefer the one with more candidates
                best_end = -1
                best_candidates = []
                
                for end in [0, 1]:
                    if end == 0:
                        current = path[0]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                    else:
                        current = path[-1]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                    
                    if len(candidates) > len(best_candidates):
                        best_end = end
                        best_candidates = candidates
                
                if best_end >= 0 and best_candidates:
                    # Pick a neighbor - prefer one that leads to more future extensions
                    # Simple heuristic: pick randomly from candidates
                    nb = rng.choice(best_candidates)
                    if (path[best_end], nb) in remaining:
                        remaining.remove((path[best_end], nb))
                    else:
                        remaining.remove((nb, path[best_end]))
                    
                    if best_end == 0:
                        path.insert(0, nb)
                    else:
                        path.append(nb)
                    changed = True
            
            paths.append(path)
        
        return paths
>>>>>>> REPLACE

