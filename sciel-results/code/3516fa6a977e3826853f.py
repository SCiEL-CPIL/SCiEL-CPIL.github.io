<<<<<<< SEARCH
    # Phase 4: polish edges for best values
    if time.time() < deadline - 3.0:
        elist = list(best_edges)
        it = 0
        while it < 30000 and time.time() < deadline - 2.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            s = score_delta(best_vals, elist, e_out, e_in)
            if s < best_score:
                elist[i] = e_in
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s
=======
    # Phase 4: polish edges for best values
    if time.time() < deadline - 3.0:
        elist = list(best_edges)
        sum_counts, prod_counts = build_counts(best_vals, elist)
        it = 0
        while it < 100000 and time.time() < deadline - 2.0: # Increased iterations due to O(1) cost
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            
            s = fast_swap_score(best_vals, elist, best_edges, sum_counts, prod_counts, i, e_in)
            if s is not None and s < best_score:
                # Apply swap
                u_out, v_out = e_out
                u_in, v_in = e_in
                s_out = best_vals[u_out] + best_vals[v_out]
                p_out = best_vals[u_out] * best_vals[v_out]
                s_in = best_vals[u_in] + best_vals[v_in]
                p_in = best_vals[u_in] * best_vals[v_in]
                
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0: del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0: del prod_counts[p_out]
                
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                elist[i] = e_in
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s
>>>>>>> REPLACE
