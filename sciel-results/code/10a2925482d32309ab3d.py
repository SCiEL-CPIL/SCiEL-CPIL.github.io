                    # Check if v's literal is currently true
                    sign_v = clause_signs[ci][np.where(clause_vars[ci] == v)[0][0]]
                    desired_v = 1 if sign_v > 0 else 0
                    is_v_lit_true = (assignment[v] == desired_v)
                    
                    if is_v_lit_true:
                        if currently_sat:
                            # Check if any other literal is true
                            # If yes, loss=0. If no, loss=1.
                            # We can check this by seeing if the clause would still be sat if v was false.
                            # This is equivalent to checking if any other lit is true.
                            other_true = False
                            for l in range(3):
                                if clause_vars[ci][l] != v:
                                    desired = 1 if clause_signs[ci][l] > 0 else 0
                                    if assignment[clause_vars[ci][l]] == desired:
                                        other_true = True
                                        break
                            if not other_true:
                                loss += 1
                    else:
                        if not currently_sat:
                            gain += 1
