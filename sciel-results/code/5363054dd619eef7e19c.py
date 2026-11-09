def priority(vector, n):
    # Heuristic: Prioritize vectors with higher Hamming weight
    # This tends to select "dense" vectors first, which may reduce
    # early conflicts compared to sparse vectors that conflict along axes.
    weight = 0
    for x in vector:
        if x != 0:
            weight += 1
    return float(weight)
