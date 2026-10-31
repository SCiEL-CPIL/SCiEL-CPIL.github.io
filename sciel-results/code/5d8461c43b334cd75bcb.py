def priority(vector, n):
    # Heuristic: Prioritize vectors with high "density" (many non-zero elements).
    # In F3, x^2 is 0 if x=0, and 1 if x=1 or x=2.
    # Sum of squares is the count of non-zero elements (Hamming weight).
    # Low-weight vectors (like 0) are "hubs" in the line graph, forming lines
    # with many other pairs. By deprioritizing them (adding them last), we allow
    # the algorithm to pick more peripheral vectors first, which may lead to a
    # larger overall cap set as hubs become invalid due to conflicts with
    # previously selected peripheral vectors.
    weight = sum(x * x for x in vector)
    return float(weight)
