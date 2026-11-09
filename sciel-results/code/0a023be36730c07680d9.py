            changes = []
            # ... inside loop ...
            # changes.append((idx_t, -1))
            # changes.append((idx_t2, 1))
            
            # ... after loop ...
            if rejected:
                for idx_c, delta_c in changes:
                    dom_count[idx_c] -= delta_c
                # in_m already reverted
