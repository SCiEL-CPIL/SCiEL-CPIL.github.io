            # Grow phase if 2-colorable
            if is_2_colorable(n, current_edges):
                # Try to add edges from the remaining pool
                remaining = all_possible[num_edges_start:]
                random.shuffle(remaining)
                for e in remaining:
                    if time.time() > deadline:
                        break
                    test_edges = current_edges + [list(e)]
                    if not is_2_colorable(n, test_edges):
                        current_edges = test_edges
                        break # Found non-2-colorable, go to prune
                else:
                    # Still 2-colorable, skip this attempt
                    continue
