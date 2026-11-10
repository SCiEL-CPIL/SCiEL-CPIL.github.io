def construct():
    random.seed(12345)
    best_pts = None
    best_score = 10**9
    best_triples = -1
    while _time_left() > 5.0:
        # start config
        mode = random.random()
        if mode < 0.3:
            # parabola
            pts = [[x, x*x] for x in range(-10,10)]
        else:
            used=set(); pts=[]
            while len(pts)<20:
                p=_random_valid_point(used); used.add(p); pts.append(p)
            if not _valid(pts):
                continue
        pts=[tuple(p) for p in pts]
        # local search to increase triples
        used=set(pts)
        cur_triples = len(_triples(pts))
        iters=0
        while _time_left()>5.0 and iters<2000:
            iters+=1
            newpts=_create_triple_move(pts, used)
            if newpts is None:
                continue
            nt=len(_triples(newpts))
            if nt>cur_triples:
                pts=newpts; used=set(pts); cur_triples=nt
            elif nt==cur_triples and random.random()<0.1:
                pts=newpts; used=set(pts)
        if _time_left()<5.0: break
        # compute exact score
        sc=_exact_score(pts)
        if sc<best_score or (sc==best_score and cur_triples>best_triples):
            best_score=sc; best_pts=pts; best_triples=cur_triples
    if best_pts is None:
        return [[x,x*x] for x in range(-10,10)]
    return [[int(x),int(y)] for (x,y) in best_pts]
