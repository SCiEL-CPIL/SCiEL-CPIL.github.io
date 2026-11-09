def fitness(A0, A1, n):
    score = 0
    for u in range(n):
        for v in range(u+1, n):
            # Check Color 0
            S0 = A0[u] & A0[v]
            S0[u] = False
            S0[v] = False
            score += count_k4(A0, S0)
            
            # Check Color 1
            S1 = A1[u] & A1[v]
            S1[u] = False
            S1[v] = False
            score += count_k4(A1, S1)
    return score

def count_k4(A, S):
    verts = np.where(S)[0]
    m = len(verts)
    if m < 4: return 0
    sub = A[np.ix_(verts, verts)]
    count = 0
    # Iterate over pairs in sub
    for i in range(m):
        for j in range(i+1, m):
            if sub[i,j]:
                common = sub[i] & sub[j]
                common[i] = False
                common[j] = False
                # Find pairs in common
                commons = np.where(common)[0]
                for k in range(len(commons)):
                    for l in range(k+1, len(commons)):
                        if sub[commons[k], commons[l]]:
                            count += 1
    # Each K4 is counted how many times?
    # A K4 on {a,b,c,d}.
    # Pairs (a,b): common neighbors c,d. Check (c,d). Count 1.
    # Pairs (a,c): common neighbors b,d. Check (b,d). Count 1.
    # ... 6 pairs in K4. Each pair sees the other 2 as common neighbors.
    # So each K4 is counted 6 times.
    return count // 6
