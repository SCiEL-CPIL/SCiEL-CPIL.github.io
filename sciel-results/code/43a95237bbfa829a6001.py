<<<<<<< SEARCH
    def schoolbook_2x2_2x1(va, vb, vc):
        for i in range(2):
            for j in range(2):
                add_term(va[i][j], vb[j], vc[i])
=======
    def schoolbook_2x2_2x1(va, vb, vc):
        for i in range(2):
            for j in range(2):
                # va is 2x2 grid of dicts
                # vb is 2x1 grid of dicts (list of lists)
                # vc is 2x1 grid of dicts
                # vb[j] is a list containing one dict
                # vc[i] is a list containing one dict
                v_entry = vb[j][0]
                w_entry = vc[i][0]
                add_term(va[i][j], v_entry, w_entry)
>>>>>>> REPLACE
