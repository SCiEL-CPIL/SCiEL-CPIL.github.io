            if step%50==0:
                # targeted: find violating ap, try to fix
                # pick a random position, check its aps
                # simpler: iterate a few random aps
                for _ in range(5):
                    ap=random.choice(aps)
                    if len({col[ap[0]],col[ap[1]],col[ap[2]],col[ap[3]]})>=3: continue
                    # try to fix by recoloring one of the 4
                    bestmove=None; bestd=0
                    for i in range(4):
                        p=ap[i]; old=col[p]
                        for c in range(k):
                            if c==old: continue
                            d=delta(col,p,c)
                            if d<bestd: bestd=d; bestmove=(p,c)
                    if bestmove and bestd<0:
                        p,c=bestmove; col[p]=c; v+=bestd
                        if v<bestv: bestv=v; best=col[:]
                        if v==0: return col
                        break
