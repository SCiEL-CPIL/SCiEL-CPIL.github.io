        if res.x is not None:
            x = res.x
            mip_set = set(i for i in range(n_vars) if x[i] > 0.5)
            if mip_set:
                # Verify validity
                valid = True
                for i in mip_set:
                    for j in mip_set:
                        if i < j and j in conflicts[i]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    mip_score = sum(weights[i] for i in mip_set)
                    if mip_score > best_score:
                        best_subset = list(mip_set)
                        best_score = mip_score
