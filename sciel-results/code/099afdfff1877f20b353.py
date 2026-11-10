# S_curr has size j+1, S_next has size j+2
# S(i, j) = S(i, j+1) * d
# S_curr[i] = S_next[i] * d
S_curr = S_next[:-1] * d
