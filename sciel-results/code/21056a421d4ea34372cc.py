            for x in S_list:
                # ... find best_y ...
                if best_y is not None:
                    # Apply
                    S.remove(x)
                    S.add(best_y)
                    W = best_new_W
                    score = best_new_score
                    improved = True
                    # Break out to restart pass with new state
                    break 
