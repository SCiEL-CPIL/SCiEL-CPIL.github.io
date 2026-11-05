                block_set = set(block)
                # ... inside loop
                if x in block_set:
                    continue
                # ... after append
                block_set.add(best_point)
