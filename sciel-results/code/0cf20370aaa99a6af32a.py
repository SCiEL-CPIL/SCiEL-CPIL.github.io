                # Find exact max
                curr_max = 0
                # Check from best_k - 1 down to 1
                # Or just check best_k - 1. If true, max >= best_k - 1.
                # If false, max < best_k - 1.
                # Let's just binary search or linear search down.
                # best_k is small.
                for s in range(best_k - 1, 0, -1):
                    if has_large_triangle_free(new_edges, s):
                        curr_max = s
                        break
                if curr_max < best_k:
                    best_k = curr_max
                    best_edges = new_edges
                    edges = new_edges
