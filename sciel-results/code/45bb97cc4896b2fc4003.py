for ci, clause in enumerate(clauses):
    for lit in clause:
        if abs(lit) - 1 == v:
            c_idx.append(ci)
            l_sign.append(1 if lit > 0 else -1)
