<<<<<<< SEARCH
    # ---------- Phase 2: Focused local search on best graph ----------
    # Try all single-edge flips that improve, iteratively
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for u, v in edge_pairs:
            if time.time() > deadline - 10:
                break
            if (best_bm >> (u * n + v)) & 1:
                cand_bm = try_remove_edge(best_bm, u, v)
            else:
                cand_bm = try_add_edge(best_bm, u, v)
            if cand_bm is None:
                continue
            cand_score = max_triangle_free_induced_bm(cand_bm, timeout=1.0)
            if cand_score < best_score:
                best_bm = cand_bm
                best_score = cand_score
                improved = True
=======
    def _targeted_local_search(bm, score, deadline):
        """Try edge additions that specifically break the largest triangle-free set."""
        _, tf_set = max_triangle_free_induced_bm(bm, timeout=2.0, return_set=True)
        if tf_set == 0:
            return bm
        
        tf_vertices = [i for i in range(n) if (tf_set >> i) & 1]
        if len(tf_vertices) <= score:
            return bm
        
        # Try adding edges between vertices in the triangle-free set
        # that don't create K4
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(tf_vertices)):
                for j in range(i + 1, len(tf_vertices)):
                    u, v = tf_vertices[i], tf_vertices[j]
                    if (bm >> (u * n + v)) & 1:
                        continue
                    cand_bm = try_add_edge(bm, u, v)
                    if cand_bm is None:
                        continue
                    cand_score = max_triangle_free_induced_bm(cand_bm, timeout=1.0)
                    if cand_score < score:
                        bm = cand_bm
                        score = cand_score
                        improved = True
                        # Update tf_set
                        _, tf_set = max_triangle_free_induced_bm(bm, timeout=1.0, return_set=True)
                        tf_vertices = [i for i in range(n) if (tf_set >> i) & 1]
                        break
                if improved:
                    break
        return bm

    # ---------- Phase 2: Focused local search on best graph ----------
    # First do targeted search, then general single-edge flips
    best_bm = _targeted_local_search(best_bm, best_score, deadline)
    best_score = max_triangle_free_induced_bm(best_bm, timeout=2.0)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for u, v in edge_pairs:
            if time.time() > deadline - 10:
                break
            if (best_bm >> (u * n + v)) & 1:
                cand_bm = try_remove_edge(best_bm, u, v)
            else:
                cand_bm = try_add_edge(best_bm, u, v)
            if cand_bm is None:
                continue
            cand_score = max_triangle_free_induced_bm(cand_bm, timeout=1.0)
            if cand_score < best_score:
                best_bm = cand_bm
                best_score = cand_score
                improved = True
                # Apply targeted search after improvement
                if time.time() < deadline - 15:
                    best_bm = _targeted_local_search(best_bm, best_score, deadline)
                    best_score = max_triangle_free_induced_bm(best_bm, timeout=1.5)
>>>>>>> REPLACE

