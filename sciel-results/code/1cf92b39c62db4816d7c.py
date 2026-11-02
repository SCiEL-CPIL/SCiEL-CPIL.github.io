s = sum(Ua[i*m+j, rr] * Va[j*p+k, rr] * Wa[k*n+i, rr] for rr in range(r))
if abs(s - 1.0) > 1e-9: return False
