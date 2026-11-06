                if best_path_in_iter and len(best_path_in_iter) > 1:
                    curr_paths.append(best_path_in_iter)
                    # Remove edges from curr_edges
                    for i in range(len(best_path_in_iter) - 1):
                        u, v = best_path_in_iter[i], best_path_in_iter[i + 1]
                        e = (min(u, v), max(u, v))
                        curr_edges.discard(e)
                    
                    # If we covered all edges, we are done with this restart
                    if not curr_edges:
                        break
