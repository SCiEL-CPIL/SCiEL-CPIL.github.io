def can_add(px, py, S_x, S_y):
    n = len(S_x)
    for i in range(n):
        qx = S_x[i]
        qy = S_y[i]
        d_pq = (px-qx)**2 + (py-qy)**2
        for j in range(i+1, n):
            rx = S_x[j]
            ry = S_y[j]
            d_pr = (px-rx)**2 + (py-ry)**2
            d_qr = (qx-rx)**2 + (qy-ry)**2
            if d_pq == d_pr or d_pq == d_qr or d_pr == d_qr:
                return False
    return True
