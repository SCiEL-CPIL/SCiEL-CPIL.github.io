def try_rainbow_bipartite(time_limit):
    import random
    n=10
    # C[i][j] colour
    best=None
    best_viol=10**9
    deadline=time.time()+time_limit
    for restart in range(...):
        C=[[random.randint(0,19) for j in range(n)] for i in range(n)]
        viol = count_violations(C)
        while viol>0 and time.time()<deadline:
            # pick a violating 4-cycle
            i1,i2,j1,j2 = random_violating_cycle(C)
            # try re-colouring each of 4 edges
            improved=False
            for (ei,ej) in [(i1,j1),(i2,j1),(i2,j2),(i1,j2)]:
                old=C[ei][ej]
                bestc=None; bestv=viol
                for c in range(20):
                    if c==old: continue
                    C[ei][ej]=c
                    v=count_violations_affected(C,ei,ej, viol)  # or full recount
                    if v<bestv:
                        bestv=v; bestc=c
                if bestc is not None:
                    C[ei][ej]=bestc; viol=bestv; improved=True; break
                else:
                    C[ei][ej]=old
            if not improved:
                break  # stuck, restart
        if viol==0:
            return C  # success
        if viol<best_viol:
            best_viol=viol; best=C
    return None
