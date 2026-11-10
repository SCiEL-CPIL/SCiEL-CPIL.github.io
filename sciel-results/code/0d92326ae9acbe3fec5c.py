def _local_search(path, deadline):
    best=np.array(path, float)
    best_vol, best_cnt = _volume(best)
    step=0.02
    t0=time.time()
    no_improve=0
    rng=np.random.default_rng(0)
    while time.time()-t0 < deadline:
        improved=False
        # coordinate descent pass
        for pi in range(len(best)):
            if time.time()-t0 > deadline: break
            for dim in range(6):
                orig=best[pi,dim]
                for delta in (step, -step):
                    cand=best.copy()
                    cand[pi,dim]=orig+delta
                    v,c=_volume(cand)
                    if v>best_vol+1e-9:
                        best=cand; best_vol=v; best_cnt=c; improved=True; no_improve=0
                        break
                else:
                    no_improve+=1
            if no_improve> len(best)*6: 
                step*=0.7; no_improve=0
        # random perturbations
        for _ in range(3):
            if time.time()-t0>deadline: break
            cand=best.copy()
            k=rng.integers(1,4)
            for _ in range(k):
                pi=rng.integers(len(best)); dim=rng.integers(6)
                cand[pi,dim]+=rng.normal(0, step*2)
            v,c=_volume(cand)
            if v>best_vol+1e-9:
                best=cand; best_vol=v; best_cnt=c; improved=True
        if not improved:
            step*=0.6
            if step<1e-4: break
    return best, best_vol
