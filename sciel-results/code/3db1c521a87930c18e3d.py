def objective(v):
    pts = v.reshape(N,2)
    pts = pts - pts.mean(axis=0)
    L = max(hex_L(p[0],p[1]) for p in pts)
    pen = 0.0
    for i in range(N):
        for j in range(i+1,N):
            d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
            if d < 2.0: pen += (2.0-d)**2
    return L + 1e5*pen
