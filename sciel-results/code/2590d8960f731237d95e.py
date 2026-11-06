            delta = 0
            for k, idx in enumerate(affected):
                old_v = old_violations[k]
                new_v = evaluate_6set(all_6sets[idx], adj)
                if new_v != old_v:
                    violations[idx] = new_v
                    delta += new_v - old_v
