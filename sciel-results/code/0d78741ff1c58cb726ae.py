def validate(U, V, W, n, m, p):
    if U.shape[0] != n*m: raise ValueError(...)
    if V.shape[0] != m*p: raise ValueError(...)
    if W.shape[0] != p*n: raise ValueError(...)
