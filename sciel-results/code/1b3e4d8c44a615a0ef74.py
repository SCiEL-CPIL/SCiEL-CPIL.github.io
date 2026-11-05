    # 2. Greedy Hitting Set (Optimized)
    def greedy_hitting_set(rng):
        covered = [False] * n_faces
        covered_count = 0
        R = set()
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        for _ in range(n_faces):
            best_e = -1
            max_ben = -1
            max_edges = []
            for e in range(n_edges):
                if e not in R:
                    if benefit[e] > max_ben:
                        max_ben = benefit[e]
                        max_edges = [e]
                    elif benefit[e] == max_ben:
                        max_edges.append(e)
            
            if not max_edges or max_ben == 0:
                break
                
            chosen_e = rng.choice(max_edges)
            R.add(chosen_e)
            
            for f in edge_faces[chosen_e]:
                if not covered[f]:
                    covered[f] = True
                    covered_count += 1
                    for e in face_edges[f]:
                        if e != chosen_e and e not in R:
                            benefit[e] -= 1
            
            if covered_count == n_faces:
                break
            if time.time() - start_time > time_limit - 10:
                break
                
        return R
