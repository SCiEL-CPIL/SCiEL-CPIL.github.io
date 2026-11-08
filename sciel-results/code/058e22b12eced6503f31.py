# m1 = (a+d)(e+h), contributes: +1 to C[0,0], +1 to C[1,1]
add_term({a_i:1, d_i:1}, {e_i:1, h_i:1}, {p00:1, p11:1})

# m2 = (c+d)e, contributes: +1 to C[0,1]
add_term({c_i:1, d_i:1}, {e_i:1}, {p01:1})

# m3 = a(f-h), contributes: +1 to C[1,0], -1 to C[1,1]
add_term({a_i:1}, {f_i:1, h_i:-1}, {p10:1, p11:-1})

# m4 = d(g-e), contributes: -1 to C[0,0], +1 to C[1,0]
add_term({d_i:1}, {g_i:1, e_i:-1}, {p00:-1, p10:1})

# m5 = (a+b)h, contributes: +1 to C[0,0]
add_term({a_i:1, b_i:1}, {h_i:1}, {p00:1})

# m6 = (c-a)(e+f), contributes: +1 to C[1,1]
add_term({c_i:1, a_i:-1}, {e_i:1, f_i:1}, {p11:1})

# m7 = (b-d)(g+h), contributes: -1 to C[0,0], +1 to C[0,1], +1 to C[1,1]
add_term({b_i:1, d_i:-1}, {g_i:1, h_i:1}, {p00:-1, p01:1, p11:1})
