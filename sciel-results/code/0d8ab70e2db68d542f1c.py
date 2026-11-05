# Pure Python check
if new_color == 0:
    for others in k4s_containing[i]:
        if colors[others[0]] == 0 and colors[others[1]] == 0 and colors[others[2]] == 0:
            delta += 1
else:
    for others in k4s_containing[i]:
        if colors[others[0]] == 0 and colors[others[1]] == 0 and colors[others[2]] == 0:
            delta -= 1
            
if new_color == 1:
    for others in k5s_containing[i]:
        if (colors[others[0]] == 1 and colors[others[1]] == 1 and colors[others[2]] == 1 and
            colors[others[3]] == 1 and colors[others[4]] == 1 and colors[others[5]] == 1 and
            colors[others[6]] == 1):
            delta += 1
else:
    for others in k5s_containing[i]:
        if (colors[others[0]] == 1 and colors[others[1]] == 1 and colors[others[2]] == 1 and
            colors[others[3]] == 1 and colors[others[4]] == 1 and colors[others[5]] == 1 and
            colors[others[6]] == 1):
            delta -= 1
