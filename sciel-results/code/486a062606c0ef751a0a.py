def _sa_run(start, deadline, seed, best):
    rng=random.Random(seed)
    S=list(start)
    n,cnt,distinct,inS=_build_state(S)
    T=1.5; Tmin=0.02
    no_improve=0
    def do_add(x):
        nonlocal n,distinct
        for y in S:
            s=x+y
            if cnt[s]==0: distinct+=1
            cnt[s]+=1
        s2=2*x
        if cnt[s2]==0: distinct+=1
        cnt[s2]+=1
        n+=1
        inS.add(x); S.append(x)
    def do_remove(x):
        nonlocal n,distinct
        for y in S:
            if y!=x:
                s=x+y
                cnt[s]-=1
                if cnt[s]==0: distinct-=1
        s2=2*x
        cnt[s2]-=1
        if cnt[s2]==0: distinct-=1
        n-=1
        inS.discard(x); S.remove(x)
    def accept(d):
        if d<=0: return True
        if T<=0: return False
        return rng.random()<math.exp(-d/T)
    step=0
    while _now()<deadline:
        conf=_conflicts(n,distinct)
        r=rng.random()
        if conf==0:
            op='add' if r<0.7 else ('swap' if r<0.9 else 'remove')
        else:
            op='remove' if r<0.45 else ('swap' if r<0.8 else 'add')
        oldc=_cost(n,distinct)
        if op=='add':
            x=rng.choice(CANDS); t=0
            while x in inS and t<30: x=rng.choice(CANDS); t+=1
            if x not in inS:
                do_add(x)
                if not accept(_cost(n,distinct)-oldc): do_remove(x)
        elif op=='remove':
            pool=[y for y in S if y not in BASE_SET]
            if pool:
                x=rng.choice(pool)
                do_remove(x)
                if not accept(_cost(n,distinct)-oldc): do_add(x)
        else: # swap
            pool=[y for y in S if y not in BASE_SET]
            if pool:
                a=rng.choice(pool)
                b=rng.choice(CANDS); t=0
                while (b in inS) and t<30: b=rng.choice(CANDS); t+=1
                if b not in inS:
                    do_remove(a); do_add(b)
                    if not accept(_cost(n,distinct)-oldc):
                        do_remove(b); do_add(a)
        # update best
        conf=_conflicts(n,distinct)
        if conf==0 and n>best[0]:
            best[0]=n; best[1]=list(S)
            no_improve=0
        else:
            no_improve+=1
        step+=1
        # cooling
        T=max(Tmin, T*0.99995)
        if no_improve>3000:
            # restart
            if best[0]>=20: break
            if rng.random()<0.5:
                S[:]=list(best[1])
            else:
                S[:]=_greedy_random(seed+step)
            n,cnt,distinct,inS=_build_state(S)
            T=1.5; no_improve=0
    return
