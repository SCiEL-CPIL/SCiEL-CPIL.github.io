        import time

        def construct():
            start_time = time.time()
            best_seq = []
            
            def check_and_add(values, sums, x):
                # Calculate new sums
                new_sums = set()
                total = x
                new_sums.add(total)
                if total in sums:
                    return None
                for a in reversed(values):
                    total += a
                    new_sums.add(total)
                    if total in sums:
                        return None
                # Check internal collisions in new_sums?
                # The sums generated are x, x+a_k, x+a_k+a_{k-1}, ...
                # Are they distinct?
                # Since values are strictly increasing positive integers,
                # the partial sums x, x+a_k, ... are strictly increasing.
                # So no internal collisions in new_sums.
                # We just need to check against existing sums.
                # Wait, the check above `if total in sums` handles that.
                # But we need to ensure new_sums don't collide with *each other*?
                # They are cumulative sums of positive numbers, so they are distinct.
                # So we just need `sums` and `new_sums` to be disjoint.
                
                # Optimization: Check intersection
                if sums.intersection(new_sums):
                    return None
                
                return new_sums

            def backtrack(last_val, values, sums):
                nonlocal best_seq
                if len(values) > len(best_seq):
                    best_seq = list(values)
                    
                # Heuristic: If we can't possibly beat best_seq, return?
                # Max remaining elements?
                # We have range [last_val+1, 256].
                # Rough bound: we can add at most (256 - last_val) elements if we pick consecutive integers,
                # but constraints are tighter.
                # Let's just rely on time limit.
                
                for x in range(last_val + 1, 257):
                    if time.time() - start_time > 85:
                        return
                    
                    new_sums = check_and_add(values, sums, x)
                    if new_sums is not None:
                        # Pruning: If we have already found a sequence of length L,
                        # and we are at depth D, can we reach L?
                        # Simple prune: if D + (256 - x) < len(best_seq): break?
                        # No, we might pick sparse numbers.
                        # But usually, smaller gaps allow more numbers.
                        # If we pick a large x, we waste space.
                        # So if x is large, and we can't beat best_seq, we might break?
                        # Let's stick to simple DFS first.
                        
                        values.append(x)
                        sums.update(new_sums)
                        
                        backtrack(x, values, sums)
                        
                        values.pop()
                        sums.difference_update(new_sums)

            backtrack(0, [], set())
            return best_seq
        