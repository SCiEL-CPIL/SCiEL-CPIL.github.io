                if conflict:
                    return False
                if satisfied:
                    continue
                if len(new_clause) == 0:
                    return False
                if len(new_clause) == 1:
                    var, sign = new_clause[0]
                    if assignment[var] == -1:
                        assignment[var] = sign
                        changed = True
                    elif assignment[var] != sign:
                        return False
                    continue
                else:
                    new_clauses.append(new_clause)
