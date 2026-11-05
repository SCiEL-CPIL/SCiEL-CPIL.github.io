            step += 1
            if step % 500 == 0 and curr_penalty == 0:
                # Quick polish
                curr_M = polish_reduce_M(curr_rows, curr_M, time.time() + 0.5)
                if curr_M < best_M:
                    best_M = curr_M
                    best_rows = copy.deepcopy(curr_rows)
