ua = sum(U[ip * m + jp, r] * A[ip, jp] for ip in range(n) for jp in range(m))
vb = sum(V[jp * p + kp, r] * B[jp, kp] for jp in range(m) for kp in range(p))
s += W[k * n + i, r] * ua * vb
