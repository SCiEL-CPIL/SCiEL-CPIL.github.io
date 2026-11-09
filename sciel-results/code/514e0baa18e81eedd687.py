            if was_sat and not is_sat:
                clause_sat[ci] -= 1
                if clause_sat[ci] == 0:
                    changes += 1
                    for v, _ in clause_vars[ci]:
                        var_unsat[v] += 1
            elif not was_sat and is_sat:
                clause_sat[ci] += 1
                if clause_sat[ci] == 1:
                    changes += 1
                    for v, _ in clause_vars[ci]:
                        var_unsat[v] -= 1
