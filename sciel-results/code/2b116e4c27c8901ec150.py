new_diffs = []
valid = True
for ai in a:
    d = abs(x - ai)
    if d == 0: continue # x in a is handled earlier
    if d in used_diffs:
        valid = False
        break
    new_diffs.append(d)
